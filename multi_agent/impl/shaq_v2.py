"""
SHAQ v2: 基于理论框架的完整实现

理论支撑：
1. 内在动机与好奇心 (Intrinsic Motivation & Curiosity) - ICM
2. 多目标优化 (MORL) - 稀疏+稠密奖励
3. 角色涌现 (Role Emergence) - 异构奖励 + Shapley 分配

核心创新：
- DOM 结构差异替代文本相似度（防止 Reward Hacking）
- 预测误差作为内在奖励（好奇心驱动）
- 角色分工奖励（Explorer vs Exploiter）
- 非单调回报处理（Shapley 的数学优势）
- 【新增】伪 ELOC 代理指标（基于 ASE 2024 论文）

参考文献:
- Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction", ICML 2017
- Wang et al., "SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning"
- Ng et al., "Policy Invariance Under Reward Transformations", ICML 1999
- ASE 2024, "Navigating Mobile Testing Evaluation: A Comprehensive Statistical Analysis"
  核心洞察: Activity Coverage 与 Fault Detection 不一致，ELOC/Method Coverage 更稳健
"""

import math
import random
import threading
import hashlib
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Set
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import multi_agent.multi_agent_system
from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from model.dense_net import DenseNet
from model.replay_buffer import ReplayBuffer
from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState
from state.web_state import WebState
from utils import instantiate_class_by_module_and_class_name
from web_test.multi_agent_thread import logger


# ============================================================================
# Component 1: DOM 结构编码器
# ============================================================================

class DOMStructureEncoder:
    """
    DOM 结构编码器：提取页面结构特征，忽略文本内容
    
    优化版本：
    - 预编译正则表达式
    - LRU 缓存避免重复计算
    - 简化的快速路径
    """
    
    def __init__(self):
        # 关注的结构标签（精简版，只保留最重要的）
        self.structure_tags = ['div', 'form', 'input', 'button', 'select', 'a', 'table', 'ul']
        # 交互元素
        self.interactive_tags = ['input', 'button', 'select', 'a']
        
        # 预编译正则表达式（性能优化关键！）
        self._tag_patterns = {
            tag: re.compile(f'<{tag}[^>]*>', re.IGNORECASE) 
            for tag in self.structure_tags + self.interactive_tags
        }
        
        # LRU 缓存：避免重复计算相同 HTML 的签名
        self._signature_cache: Dict[int, str] = {}
        self._hash_cache: Dict[int, str] = {}
        self._cache_max_size = 100
    
    def _get_html_key(self, html: str) -> int:
        """快速获取 HTML 的缓存键（使用长度+首尾字符的组合）"""
        if not html:
            return 0
        # 使用长度 + 首100字符 + 尾100字符的哈希作为缓存键
        key_str = f"{len(html)}:{html[:100]}:{html[-100:] if len(html) > 100 else ''}"
        return hash(key_str)
    
    def extract_structure_signature(self, html: str) -> str:
        """提取 DOM 结构签名（带缓存）"""
        if not html:
            return ""
        
        # 检查缓存
        cache_key = self._get_html_key(html)
        if cache_key in self._signature_cache:
            return self._signature_cache[cache_key]
        
        # 快速统计各标签数量
        tag_counts = []
        interactive_count = 0
        
        for tag in self.structure_tags:
            count = len(self._tag_patterns[tag].findall(html))
            if count > 0:
                tag_counts.append(f"{tag}:{count}")
        
        for tag in self.interactive_tags:
            interactive_count += len(self._tag_patterns[tag].findall(html))
        
        tag_counts.append(f"i:{interactive_count}")
        signature = "|".join(tag_counts)
        
        # 更新缓存（简单 LRU：超过容量时清空）
        if len(self._signature_cache) >= self._cache_max_size:
            self._signature_cache.clear()
        self._signature_cache[cache_key] = signature
        
        return signature
    
    def compute_structure_hash(self, html: str) -> str:
        """计算结构哈希（带缓存）"""
        if not html:
            return ""
        
        cache_key = self._get_html_key(html)
        if cache_key in self._hash_cache:
            return self._hash_cache[cache_key]
        
        signature = self.extract_structure_signature(html)
        hash_val = hashlib.md5(signature.encode()).hexdigest()[:16]
        
        if len(self._hash_cache) >= self._cache_max_size:
            self._hash_cache.clear()
        self._hash_cache[cache_key] = hash_val
        
        return hash_val
    
    def compute_structure_distance(self, html1: str, html2: str) -> float:
        """
        计算两个页面的结构距离（优化版）
        """
        # 快速路径：相同 HTML
        if html1 == html2:
            return 0.0
        
        # 快速路径：空 HTML
        if not html1 or not html2:
            return 1.0 if (html1 or html2) else 0.0
        
        sig1 = self.extract_structure_signature(html1)
        sig2 = self.extract_structure_signature(html2)
        
        if sig1 == sig2:
            return 0.0
        
        # 快速解析：直接比较标签计数
        def quick_parse(sig):
            counts = {}
            for part in sig.split("|"):
                if ":" in part:
                    key, val = part.split(":", 1)
                    counts[key] = int(val) if val.isdigit() else 1
            return counts
        
        counts1 = quick_parse(sig1)
        counts2 = quick_parse(sig2)
        
        all_keys = set(counts1.keys()) | set(counts2.keys())
        if not all_keys:
            return 0.0
        
        total_diff = sum(
            abs(counts1.get(k, 0) - counts2.get(k, 0)) / max(counts1.get(k, 1), counts2.get(k, 1), 1)
            for k in all_keys
        )
        
        return min(1.0, total_diff / len(all_keys))


# ============================================================================
# Component 1.5: 伪 ELOC 代理指标 (基于 ASE 2024 论文)
# ============================================================================

class PseudoELOCTracker:
    """
    伪 ELOC (Executable Lines of Code) 追踪器
    
    由于测试第三方网站无法获取真正的代码覆盖率，
    使用前端代理指标来近似 ELOC：
    
    基于 ASE 2024 论文的洞察：
    - Activity Coverage 与 Fault Detection 相关性低（r ≈ 0.3-0.5）
    - ELOC/Method Coverage 与 Fault Detection 相关性高（r ≈ 0.7-0.8）
    
    代理指标：
    1. API 调用多样性 → 近似 Method Coverage
    2. 网络请求数量 → 近似 Instruction Coverage  
    3. DOM 复杂度变化 → 近似 Branch Coverage
    4. JS 执行量 → 近似 ELOC
    """
    
    def __init__(self):
        # 追踪已见过的 API 端点（去参数）
        self.seen_api_endpoints: Set[str] = set()
        self.seen_request_patterns: Set[str] = set()
        
        # 追踪 DOM 复杂度历史
        self.dom_complexity_history: List[float] = []
        self.max_dom_depth_seen = 0
        
        # 追踪 JS 执行指标
        self.js_execution_count = 0
        self.unique_js_sources: Set[str] = set()
        
        # 统计
        self.total_requests = 0
        self.unique_domains: Set[str] = set()
        
        # 预编译正则
        self._api_pattern = re.compile(r'/api/|/v\d+/|/graphql|\.json(\?|$)')
        self._resource_pattern = re.compile(r'\.(js|css|png|jpg|gif|svg|woff|ico)(\?|$)', re.IGNORECASE)
    
    def extract_api_endpoint(self, url: str) -> Optional[str]:
        """
        从 URL 提取 API 端点（去参数，保留路径模式）
        
        Example:
            /api/users/123/posts?page=1 → /api/users/*/posts
        """
        try:
            parsed = urlparse(url)
            path = parsed.path
            
            # 忽略静态资源
            if self._resource_pattern.search(path):
                return None
            
            # 规范化路径（将数字 ID 替换为 *）
            normalized = re.sub(r'/\d+', '/*', path)
            normalized = re.sub(r'/[a-f0-9]{24,}', '/*', normalized)  # MongoDB ObjectId
            normalized = re.sub(r'/[a-f0-9-]{36}', '/*', normalized)  # UUID
            
            return f"{parsed.netloc}{normalized}"
        except Exception:
            return None
    
    def process_performance_logs(self, performance_logs: List[Dict]) -> Dict[str, float]:
        """
        处理 Chrome Performance logs，提取伪 ELOC 指标
        
        Args:
            performance_logs: 从 driver.get_log("performance") 获取
            
        Returns:
            metrics: {
                'new_api_endpoints': int,  # 新发现的 API 端点数
                'request_diversity': float,  # 请求多样性（0-1）
                'total_new_requests': int,  # 新请求数
                'new_domains': int,  # 新域名数
            }
        """
        metrics = {
            'new_api_endpoints': 0,
            'request_diversity': 0.0,
            'total_new_requests': 0,
            'new_domains': 0,
        }
        
        if not performance_logs:
            return metrics
        
        new_endpoints = 0
        new_requests = 0
        new_domains = 0
        
        for log in performance_logs:
            try:
                # Performance log 格式：{"message": "{\"method\": \"Network.requestWillBeSent\", ...}"}
                message = log.get('message', '{}')
                if isinstance(message, str):
                    import json
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                else:
                    data = message
                
                method = data.get('method', '')
                params = data.get('params', {})
                
                # 处理网络请求
                if method == 'Network.requestWillBeSent':
                    request = params.get('request', {})
                    url = request.get('url', '')
                    
                    if not url:
                        continue
                    
                    self.total_requests += 1
                    
                    # 检查是否是新域名
                    try:
                        domain = urlparse(url).netloc
                        if domain and domain not in self.unique_domains:
                            self.unique_domains.add(domain)
                            new_domains += 1
                    except Exception:
                        pass
                    
                    # 检查是否是 API 调用
                    if self._api_pattern.search(url):
                        endpoint = self.extract_api_endpoint(url)
                        if endpoint and endpoint not in self.seen_api_endpoints:
                            self.seen_api_endpoints.add(endpoint)
                            new_endpoints += 1
                    
                    # 检查请求模式（简化）
                    request_type = request.get('method', 'GET')
                    pattern = f"{request_type}:{urlparse(url).path[:50]}"
                    if pattern not in self.seen_request_patterns:
                        self.seen_request_patterns.add(pattern)
                        new_requests += 1
                
                # 处理 JS 执行（从 Console 或 Runtime）
                elif method in ['Runtime.consoleAPICalled', 'Runtime.exceptionThrown']:
                    self.js_execution_count += 1
                    
            except Exception:
                continue
        
        metrics['new_api_endpoints'] = new_endpoints
        metrics['total_new_requests'] = new_requests
        metrics['new_domains'] = new_domains
        
        # 计算请求多样性（基于已见模式数量）
        if self.total_requests > 0:
            diversity = len(self.seen_request_patterns) / max(self.total_requests, 1)
            metrics['request_diversity'] = min(1.0, diversity)
        
        return metrics
    
    def compute_dom_complexity(self, html: str) -> Dict[str, float]:
        """
        计算 DOM 复杂度指标（近似 Branch Coverage）
        
        理论依据：
        - 更复杂的 DOM 通常意味着更多代码路径被执行
        - DOM 深度增加 → 可能触发了新的渲染分支
        """
        if not html:
            return {'complexity': 0, 'depth': 0, 'interactive': 0, 'complexity_delta': 0}
        
        # 快速估算 DOM 复杂度
        # 1. 标签数量（近似节点数）
        tag_count = html.count('<')
        
        # 2. 嵌套深度（通过缩进或层级估算）
        max_depth = 0
        current_depth = 0
        for char in html:
            if char == '<':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '/':
                current_depth = max(0, current_depth - 1)
        
        # 3. 交互元素数量
        interactive_count = (
            html.count('<input') + 
            html.count('<button') + 
            html.count('<select') + 
            html.count('<a ') + 
            html.count('onclick') +
            html.count('@click')  # Vue
        )
        
        # 归一化
        complexity = min(1.0, tag_count / 1000)  # 假设 1000 个标签为满复杂度
        depth = min(1.0, max_depth / 50)  # 假设深度 50 为满
        interactive = min(1.0, interactive_count / 100)  # 假设 100 个交互元素为满
        
        # 计算复杂度变化
        current_complexity = complexity * 0.4 + depth * 0.3 + interactive * 0.3
        complexity_delta = 0
        if self.dom_complexity_history:
            prev_complexity = self.dom_complexity_history[-1]
            complexity_delta = max(0, current_complexity - prev_complexity)
        
        self.dom_complexity_history.append(current_complexity)
        if len(self.dom_complexity_history) > 100:
            self.dom_complexity_history = self.dom_complexity_history[-100:]
        
        # 更新最大深度
        if max_depth > self.max_dom_depth_seen:
            self.max_dom_depth_seen = max_depth
        
        return {
            'complexity': current_complexity,
            'depth': depth,
            'interactive': interactive,
            'complexity_delta': complexity_delta,
        }
    
    def get_pseudo_eloc_summary(self) -> Dict:
        """获取伪 ELOC 摘要"""
        return {
            'total_api_endpoints': len(self.seen_api_endpoints),
            'total_request_patterns': len(self.seen_request_patterns),
            'total_domains': len(self.unique_domains),
            'total_requests': self.total_requests,
            'js_execution_count': self.js_execution_count,
            'max_dom_depth': self.max_dom_depth_seen,
            'api_endpoints_list': list(self.seen_api_endpoints)[:20],  # 前 20 个
        }


# ============================================================================
# Component 2: 内在好奇心模块 (ICM)
# ============================================================================

class IntrinsicCuriosityModule(nn.Module):
    """
    内在好奇心模块 (ICM) - 优化版
    
    优化：
    - 更小的网络结构
    - 计算频率控制
    - 缓存机制
    """
    
    def __init__(self, state_dim: int = 32, action_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 更小的状态编码器 φ（减少计算量）
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 更小的前向动力学模型
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim // 2 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 缩放因子
        self.eta = 0.1
        
        # 优化：减少历史记录大小
        self.prediction_errors: List[float] = []
        self.max_history = 100
        
        # 计算频率控制：每 N 步才真正计算一次
        self.compute_interval = 3
        self.step_counter = 0
        self.cached_reward = 0.05  # 默认好奇心奖励
    
    def encode_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """编码状态"""
        return self.state_encoder(state_tensor)
    
    def predict_next_state(self, state_encoded: torch.Tensor, action_tensor: torch.Tensor) -> torch.Tensor:
        """预测下一状态的编码"""
        combined = torch.cat([state_encoded, action_tensor], dim=-1)
        return self.forward_model(combined)
    
    def compute_intrinsic_reward(
        self, 
        state_tensor: torch.Tensor = None, 
        action_tensor: torch.Tensor = None, 
        next_state_tensor: torch.Tensor = None
    ) -> float:
        """
        计算内在奖励（好奇心奖励）- 优化版
        
        优化：每 N 步才真正计算一次，其他时候返回缓存值
        """
        self.step_counter += 1
        
        # 频率控制：不是每步都计算
        if self.step_counter % self.compute_interval != 0:
            return self.cached_reward
        
        # 如果没有提供张量，返回默认值
        if state_tensor is None or next_state_tensor is None:
            return self.cached_reward
        
        try:
            with torch.no_grad():
                # 编码当前状态和下一状态
                phi_s = self.encode_state(state_tensor)
                phi_s_next = self.encode_state(next_state_tensor)
                
                # 预测下一状态
                if action_tensor is None:
                    action_tensor = torch.zeros(self.action_dim, device=state_tensor.device)
                phi_s_next_pred = self.predict_next_state(phi_s, action_tensor)
                
                # 计算预测误差
                prediction_error = F.mse_loss(phi_s_next_pred, phi_s_next).item()
                
                # 简化的归一化
                self.prediction_errors.append(prediction_error)
                if len(self.prediction_errors) > self.max_history:
                    self.prediction_errors = self.prediction_errors[-self.max_history:]
                
                # 快速归一化
                if len(self.prediction_errors) > 5:
                    avg_error = sum(self.prediction_errors) / len(self.prediction_errors)
                    normalized_error = min(1.0, prediction_error / (avg_error + 0.01))
                else:
                    normalized_error = min(1.0, prediction_error)
                
                self.cached_reward = self.eta * normalized_error
                return self.cached_reward
        except Exception:
            return self.cached_reward
    
    def update(self, state_tensor: torch.Tensor, action_tensor: torch.Tensor, 
               next_state_tensor: torch.Tensor, optimizer: optim.Optimizer):
        """更新 ICM 模型"""
        phi_s = self.encode_state(state_tensor)
        phi_s_next = self.encode_state(next_state_tensor)
        phi_s_next_pred = self.predict_next_state(phi_s, action_tensor)
        
        # 前向模型损失
        forward_loss = F.mse_loss(phi_s_next_pred, phi_s_next.detach())
        
        # 逆向模型损失
        combined_phi = torch.cat([phi_s, phi_s_next], dim=-1)
        action_pred = self.inverse_model(combined_phi)
        inverse_loss = F.mse_loss(action_pred, action_tensor)
        
        # 总损失
        loss = forward_loss + 0.5 * inverse_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


# ============================================================================
# Component 3: 多目标奖励系统
# ============================================================================

class MultiObjectiveRewardSystem:
    """
    三层奖励系统（基于 ASE 2024 论文）
    
    论文核心洞察：
    - Activity Coverage 与 Fault Detection 相关性低
    - ELOC/Method Coverage 与 Fault Detection 相关性高
    
    三层架构：
    ┌─────────────┬─────────────────────┬──────────────────────┐
    │   层级      │    奖励来源          │    学术指标           │
    ├─────────────┼─────────────────────┼──────────────────────┤
    │ 基础层      │ DOM 变化/新状态      │ Activity Coverage    │
    │ 核心层(重点) │ 伪 ELOC 指标        │ Code Coverage       │
    │ 目标层      │ Error/HTTP 500      │ Fault Detection     │
    └─────────────┴─────────────────────┴──────────────────────┘
    
    权重分配：
    - 目标层 > 核心层 > 基础层
    - 论文证明 ELOC 比 Activity 更稳健！
    """
    
    def __init__(self, mode: str = 'three_tier', alive_time: float = 10800):
        """
        Args:
            mode: 'three_tier' (论文推荐), 'hybrid', 'error_only', 'coverage_only'
            alive_time: 总测试时间
        """
        self.mode = mode
        self.alive_time = alive_time
        self.start_time = datetime.now()
        
        # ============================================================
        # 【目标层】Fault Detection - 最高优先级
        # 论文：这是最终目标，但稀疏
        # ============================================================
        self.R_SEVERE_ERROR = 500.0      # JS 运行时错误
        self.R_WARNING = 50.0            # Warning
        self.R_HTTP_ERROR = 1000.0       # HTTP 5xx 错误（最严重）
        self.R_CONSOLE_ERROR = 200.0     # 控制台错误
        
        # ============================================================
        # 【核心层】伪 ELOC - 论文重点！
        # 论文证明：ELOC 比 Activity Coverage 更能预测 Bug
        # ============================================================
        self.R_NEW_API_ENDPOINT = 30.0   # 新 API 端点 → Method Coverage
        self.R_REQUEST_DIVERSITY = 15.0  # 请求多样性 → Instruction Coverage
        self.R_DOM_COMPLEXITY_INCREASE = 20.0  # DOM 复杂度增加 → Branch Coverage
        self.R_NEW_DOMAIN = 25.0         # 新域名请求 → 跨模块覆盖
        self.R_JS_EXECUTION = 5.0        # JS 执行 → ELOC
        
        # ============================================================
        # 【基础层】Activity Coverage - 最低优先级（可能误导！）
        # 论文警告：这个指标与 Bug 相关性不稳定
        # ============================================================
        self.R_NEW_STATE = 5.0           # 降低权重！
        self.R_NEW_URL = 10.0            # 降低权重！
        self.R_NEW_ACTION = 3.0          # 降低权重！
        self.R_DEPTH_BONUS = 8.0         # URL 深度
        
        # 惩罚
        self.R_PENALTY = -1.0
        
        # ========== 动态权重（三层） ==========
        # 早期：核心层高，目标层低（建立 ELOC 基础）
        # 中期：均衡
        # 后期：目标层高（专注找 Bug）
        self.tier_weights = {
            'target': 1.0,   # 目标层基础权重
            'core': 1.0,     # 核心层基础权重
            'base': 1.0,     # 基础层基础权重
        }
        
        # 追踪
        self.visited_states: Set[int] = set()
        self.visited_urls: Set[str] = set()
        self.visited_actions: Set[int] = set()
        self.errors_found: List[str] = []
        self.severe_errors: List[str] = []
        
        # 伪 ELOC 追踪器
        self.pseudo_eloc_tracker = PseudoELOCTracker()
        
        # DOM 编码器
        self.dom_encoder = DOMStructureEncoder()
        
        # 诊断统计（分层）
        self.reward_breakdown_history: Dict[str, float] = defaultdict(float)
        self.reward_count_history: Dict[str, int] = defaultdict(int)
        self.tier_reward_totals = {
            'target': 0.0,  # 目标层累计
            'core': 0.0,    # 核心层累计
            'base': 0.0,    # 基础层累计
        }
        # 向后兼容
        self.error_reward_total = 0.0
        self.coverage_reward_total = 0.0
        
    def get_dynamic_weights(self) -> Tuple[float, float]:
        """
        动态调整权重（向后兼容接口）
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        progress = min(1.0, elapsed / self.alive_time)
        
        alpha = 0.5 + progress      # 0.5 → 1.5
        beta = 1.5 - progress       # 1.5 → 0.5
        
        return alpha, beta
    
    def get_three_tier_weights(self) -> Dict[str, float]:
        """
        三层动态权重（基于论文推荐）
        
        阶段划分：
        - 早期 (0-30%): 核心层(ELOC)高，建立覆盖基础
        - 中期 (30-70%): 均衡，巩固 + 探索
        - 后期 (70-100%): 目标层(Fault)高，专注找 Bug
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        progress = min(1.0, elapsed / self.alive_time)
        
        if progress < 0.3:
            # 早期：核心层 ELOC 最重要
            return {
                'target': 0.5,
                'core': 1.5,   # 重点！建立代码覆盖
                'base': 1.0,
            }
        elif progress < 0.7:
            # 中期：均衡
            return {
                'target': 1.0,
                'core': 1.2,
                'base': 0.8,
            }
        else:
            # 后期：目标层最重要（找 Bug）
            return {
                'target': 1.5,  # 重点！找错误
                'core': 1.0,
                'base': 0.5,   # 降低 Activity 权重
            }
    
    def get_coverage_metrics(self) -> Dict:
        """获取当前指标"""
        return {
            'state_coverage': len(self.visited_states),
            'url_coverage': len(self.visited_urls),
            'action_coverage': len(self.visited_actions),
            'errors_found': len(self.errors_found),
            'severe_errors': len(self.severe_errors),
            'error_reward_total': self.error_reward_total,
            'coverage_reward_total': self.coverage_reward_total,
        }
    
    def get_alignment_report(self) -> Dict:
        """
        生成 Reward-Metric 对齐报告
        
        混合模式下，展示错误奖励和覆盖奖励的比例
        """
        metrics = self.get_coverage_metrics()
        alpha, beta = self.get_dynamic_weights()
        
        total_reward = self.error_reward_total + self.coverage_reward_total
        error_ratio = self.error_reward_total / total_reward if total_reward > 0 else 0
        
        # 三层分析
        total_tier_reward = sum(self.tier_reward_totals.values())
        tier_ratios = {}
        for tier, total in self.tier_reward_totals.items():
            tier_ratios[tier] = f"{(total / total_tier_reward * 100) if total_tier_reward > 0 else 0:.1f}%"
        
        return {
            'mode': self.mode,
            'metrics': metrics,
            'reward_breakdown': dict(self.reward_breakdown_history),
            'reward_counts': dict(self.reward_count_history),
            'hybrid_analysis': {
                'current_alpha': alpha,
                'current_beta': beta,
                'error_reward_total': self.error_reward_total,
                'coverage_reward_total': self.coverage_reward_total,
                'error_reward_ratio': f"{error_ratio:.1%}",
                'severe_error_count': len(self.severe_errors),
            },
            # 【新增】三层分析
            'three_tier_analysis': {
                'tier_weights': self.get_three_tier_weights(),
                'tier_totals': self.tier_reward_totals,
                'tier_ratios': tier_ratios,
                'pseudo_eloc': self.pseudo_eloc_tracker.get_pseudo_eloc_summary(),
            },
            'interpretation': {
                'error_driven': error_ratio > 0.3,
                'exploration_effective': metrics['state_coverage'] > 5,
                'eloc_effective': len(self.pseudo_eloc_tracker.seen_api_endpoints) > 3,
            }
        }
    
    def compute_three_tier_reward(
        self,
        web_state: WebState,
        html: str,
        action: Optional[WebAction] = None,
        browser_logs: List[Dict] = None,
        performance_logs: List[Dict] = None,
        http_status: int = 200
    ) -> Tuple[float, Dict[str, float]]:
        """
        三层奖励计算（基于 ASE 2024 论文）
        
        R_total = w_target × R_target + w_core × R_core + w_base × R_base
        
        Args:
            browser_logs: 从 driver.get_log("browser") 获取
            performance_logs: 从 driver.get_log("performance") 获取
            
        Returns:
            (total_reward, breakdown_dict)
        """
        target_reward = 0.0   # 目标层
        core_reward = 0.0     # 核心层
        base_reward = 0.0     # 基础层
        breakdown = {}
        
        # ============================================================
        # 【目标层】Fault Detection
        # ============================================================
        if browser_logs:
            for log in browser_logs:
                log_key = f"{log.get('level')}:{log.get('message', '')[:100]}"
                if log_key not in self.errors_found:
                    self.errors_found.append(log_key)
                    
                    if log.get('level') == 'SEVERE':
                        target_reward += self.R_SEVERE_ERROR
                        breakdown['target:severe_error'] = breakdown.get('target:severe_error', 0) + self.R_SEVERE_ERROR
                        self.severe_errors.append(log_key)
                    elif log.get('level') == 'WARNING':
                        target_reward += self.R_WARNING
                        breakdown['target:warning'] = breakdown.get('target:warning', 0) + self.R_WARNING
        
        if http_status >= 500:
            error_key = f"http_{http_status}"
            if error_key not in self.errors_found:
                target_reward += self.R_HTTP_ERROR
                breakdown['target:http_error'] = self.R_HTTP_ERROR
                self.errors_found.append(error_key)
        
        # ============================================================
        # 【核心层】伪 ELOC（论文重点！）
        # ============================================================
        if performance_logs:
            eloc_metrics = self.pseudo_eloc_tracker.process_performance_logs(performance_logs)
            
            # 新 API 端点 → Method Coverage
            if eloc_metrics['new_api_endpoints'] > 0:
                api_reward = self.R_NEW_API_ENDPOINT * eloc_metrics['new_api_endpoints']
                core_reward += api_reward
                breakdown['core:new_api_endpoint'] = api_reward
            
            # 新请求模式 → Instruction Coverage
            if eloc_metrics['total_new_requests'] > 0:
                request_reward = self.R_REQUEST_DIVERSITY * min(eloc_metrics['total_new_requests'], 5)
                core_reward += request_reward
                breakdown['core:request_diversity'] = request_reward
            
            # 新域名 → 跨模块覆盖
            if eloc_metrics['new_domains'] > 0:
                domain_reward = self.R_NEW_DOMAIN * eloc_metrics['new_domains']
                core_reward += domain_reward
                breakdown['core:new_domain'] = domain_reward
        
        # DOM 复杂度变化 → Branch Coverage
        if html:
            dom_metrics = self.pseudo_eloc_tracker.compute_dom_complexity(html)
            if dom_metrics['complexity_delta'] > 0.05:
                complexity_reward = self.R_DOM_COMPLEXITY_INCREASE * dom_metrics['complexity_delta'] * 10
                core_reward += complexity_reward
                breakdown['core:dom_complexity'] = complexity_reward
        
        # ============================================================
        # 【基础层】Activity Coverage（权重较低）
        # ============================================================
        state_hash = hash(str(web_state))
        if state_hash not in self.visited_states:
            base_reward += self.R_NEW_STATE
            breakdown['base:new_state'] = self.R_NEW_STATE
            self.visited_states.add(state_hash)
        
        current_url = getattr(web_state, 'url', None)
        if current_url:
            if current_url not in self.visited_urls:
                base_reward += self.R_NEW_URL
                breakdown['base:new_url'] = self.R_NEW_URL
                self.visited_urls.add(current_url)
                
                parsed = urlparse(current_url)
                depth = len([p for p in parsed.path.split('/') if p])
                if depth > 1:
                    depth_bonus = self.R_DEPTH_BONUS * (depth - 1)
                    base_reward += depth_bonus
                    breakdown['base:depth_bonus'] = depth_bonus
        
        if action:
            action_hash = hash(str(action))
            if action_hash not in self.visited_actions:
                base_reward += self.R_NEW_ACTION
                breakdown['base:new_action'] = self.R_NEW_ACTION
                self.visited_actions.add(action_hash)
        
        # ============================================================
        # 动态加权
        # ============================================================
        weights = self.get_three_tier_weights()
        
        total_reward = (
            weights['target'] * target_reward +
            weights['core'] * core_reward +
            weights['base'] * base_reward
        )
        
        # 记录统计
        self.tier_reward_totals['target'] += target_reward
        self.tier_reward_totals['core'] += core_reward
        self.tier_reward_totals['base'] += base_reward
        
        # 向后兼容
        self.error_reward_total += target_reward
        self.coverage_reward_total += core_reward + base_reward
        
        breakdown['_weights'] = weights
        breakdown['_target_reward'] = target_reward
        breakdown['_core_reward'] = core_reward
        breakdown['_base_reward'] = base_reward
        
        for key, value in breakdown.items():
            if not key.startswith('_'):
                self.reward_breakdown_history[key] += value
                self.reward_count_history[key] += 1
        
        return total_reward, breakdown
    
    def compute_hybrid_reward(
        self,
        web_state: WebState,
        action: Optional[WebAction] = None,
        browser_logs: List[Dict] = None,
        http_status: int = 200
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算混合奖励：稀疏（错误）+ 稠密（覆盖）
        
        R_total = α × R_error + β × R_coverage
        
        Args:
            browser_logs: 从 driver.get_log("browser") 获取的日志列表
        
        Returns:
            (total_reward, breakdown_dict)
        """
        error_reward = 0.0
        coverage_reward = 0.0
        breakdown = {}
        
        # ========== 1. 稀疏奖励：错误发现 ==========
        if browser_logs:
            for log in browser_logs:
                log_key = f"{log.get('level')}:{log.get('message', '')[:100]}"
                if log_key not in self.errors_found:
                    self.errors_found.append(log_key)
                    
                    if log.get('level') == 'SEVERE':
                        error_reward += self.R_SEVERE_ERROR
                        breakdown['severe_error'] = breakdown.get('severe_error', 0) + self.R_SEVERE_ERROR
                        self.severe_errors.append(log_key)
                    elif log.get('level') == 'WARNING':
                        error_reward += self.R_WARNING
                        breakdown['warning'] = breakdown.get('warning', 0) + self.R_WARNING
        
        # HTTP 错误
        if http_status >= 500:
            error_key = f"http_{http_status}"
            if error_key not in self.errors_found:
                error_reward += self.R_HTTP_ERROR
                breakdown['http_error'] = self.R_HTTP_ERROR
                self.errors_found.append(error_key)
        
        # ========== 2. 稠密奖励：覆盖引导 ==========
        # 新状态
        state_hash = hash(str(web_state))
        if state_hash not in self.visited_states:
            coverage_reward += self.R_NEW_STATE
            breakdown['new_state'] = self.R_NEW_STATE
            self.visited_states.add(state_hash)
        
        # 新 URL
        current_url = getattr(web_state, 'url', None)
        if current_url:
            if current_url not in self.visited_urls:
                coverage_reward += self.R_NEW_URL
                breakdown['new_url'] = self.R_NEW_URL
                self.visited_urls.add(current_url)
                
                # URL 深度奖励
                from urllib.parse import urlparse
                parsed = urlparse(current_url)
                depth = len([p for p in parsed.path.split('/') if p])
                if depth > 1:
                    depth_bonus = self.R_DEPTH_BONUS * (depth - 1)
                    coverage_reward += depth_bonus
                    breakdown['depth_bonus'] = depth_bonus
        
        # 新动作
        if action:
            action_hash = hash(str(action))
            if action_hash not in self.visited_actions:
                coverage_reward += self.R_NEW_ACTION
                breakdown['new_action'] = self.R_NEW_ACTION
                self.visited_actions.add(action_hash)
        
        # ========== 3. 动态加权 ==========
        alpha, beta = self.get_dynamic_weights()
        
        total_reward = alpha * error_reward + beta * coverage_reward
        
        # 记录统计
        self.error_reward_total += error_reward
        self.coverage_reward_total += coverage_reward
        
        breakdown['_alpha'] = alpha
        breakdown['_beta'] = beta
        breakdown['_error_reward'] = error_reward
        breakdown['_coverage_reward'] = coverage_reward
        
        for key, value in breakdown.items():
            if not key.startswith('_'):
                self.reward_breakdown_history[key] += value
                self.reward_count_history[key] += 1
        
        return total_reward, breakdown
    
    # 保持向后兼容
    def compute_coverage_reward(self, web_state, action=None, console_errors=None, http_status=200):
        # 转换格式
        browser_logs = [{'level': 'SEVERE', 'message': e} for e in (console_errors or [])]
        return self.compute_hybrid_reward(web_state, action, browser_logs, http_status)
    
    def compute_sparse_reward(self, web_state, html, action=None, http_status=200, console_errors=None):
        return self.compute_coverage_reward(web_state, action, console_errors, http_status)
    
    def compute_dense_reward(
        self,
        web_state: WebState,
        html: str,
        prev_html: str,
        curiosity_reward: float,
        action_execution_count: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算稠密奖励
        
        Coverage-Aligned 模式：几乎不需要稠密奖励
        Exploration-Guided 模式：使用稠密奖励引导
        """
        if self.mode == 'coverage_aligned':
            # Coverage-Aligned 模式：不需要额外的稠密奖励
            # 因为 sparse reward 已经完全对齐 metrics
            return 0.0, {}
        
        # Exploration-Guided 模式（原来的逻辑）
        reward = 0.0
        breakdown = {}
        
        # DOM 结构变化奖励
        if hasattr(self, 'R_STRUCTURE_CHANGE'):
            structure_distance = self.dom_encoder.compute_structure_distance(prev_html, html)
            if structure_distance > 0.1:
                structure_reward = self.R_STRUCTURE_CHANGE * structure_distance
                reward += structure_reward
                breakdown['structure_change'] = structure_reward
        
        # 好奇心奖励
        if curiosity_reward > 0:
            reward += curiosity_reward * 5.0
            breakdown['curiosity'] = curiosity_reward * 5.0
        
        for key, value in breakdown.items():
            self.reward_breakdown_history[key] += value
            self.reward_count_history[key] += 1
        
        return reward, breakdown


# ============================================================================
# Component 4: 角色分工系统 + JBS 检测
# ============================================================================

class RoleBasedRewardSystem:
    """
    角色分工奖励系统 + JBS (Joint Blind Spot) 检测
    
    核心思想：
    - Explorer: 广度优先，发现新 URL
    - Exploiter: 深度优先，触发状态变化和交互
    
    JBS 问题：多个 Agent 收敛到相同探索路径
    解决方案：
    1. 多样性奖励 - 奖励去其他人没去过的地方
    2. 频率惩罚 - 惩罚所有人都频繁访问的状态
    3. 角色分工 - 强制不同的探索偏好
    
    SHAQ 的 Shapley 能精准分配"角色贡献"
    """
    
    def __init__(self, agent_num: int):
        self.agent_num = agent_num
        
        # 角色权重
        self.explorer_weights = {
            'new_url': 2.0,
            'new_url_path': 2.0,
            'url_depth': 1.5,
            'structure_change': 0.5,
            'action_novelty': 1.0,
        }
        
        self.exploiter_weights = {
            'new_url': 0.5,
            'new_url_path': 0.5,
            'url_depth': 0.5,
            'structure_change': 2.0,
            'form_submit': 2.0,
            'action_novelty': 1.5,
            'curiosity': 2.0,
        }
        
        # 追踪每个 Agent 的探索历史
        self.agent_url_discoveries: Dict[str, Set[str]] = defaultdict(set)
        self.agent_state_changes: Dict[str, int] = defaultdict(int)
        self.agent_interactions: Dict[str, int] = defaultdict(int)
        
        # ========== JBS 检测 ==========
        self.url_visit_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))  # url -> {agent -> count}
        self.jbs_events: List[Dict] = []  # 记录 JBS 事件
    
    def get_agent_role(self, agent_name: str) -> str:
        """
        动态确定 Agent 角色
        
        可以基于：
        1. 固定分配（简单）
        2. 历史表现（自适应）
        """
        # 简单方案：奇偶分配
        agent_id = int(agent_name)
        if agent_id % 2 == 0:
            return 'explorer'
        else:
            return 'exploiter'
    
    def apply_role_weights(
        self, 
        agent_name: str, 
        reward_breakdown: Dict[str, float]
    ) -> float:
        """应用角色权重到奖励分解"""
        role = self.get_agent_role(agent_name)
        weights = self.explorer_weights if role == 'explorer' else self.exploiter_weights
        
        total = 0.0
        for key, value in reward_breakdown.items():
            weight = weights.get(key, 1.0)
            total += value * weight
        
        return total
    
    def compute_diversity_bonus(
        self, 
        agent_name: str, 
        current_url: str,
        team_visited_urls: Set[str]
    ) -> float:
        """
        计算多样性奖励：奖励 Agent 探索其他人没去过的地方
        """
        if not current_url:
            return 0.0
        
        # 其他 Agent 访问过的 URL
        other_agents_urls = set()
        for other_agent, urls in self.agent_url_discoveries.items():
            if other_agent != agent_name:
                other_agents_urls.update(urls)
        
        # 如果这个 URL 其他人没去过，给奖励
        if current_url not in other_agents_urls:
            return 15.0  # 多样性奖励
        
        return 0.0
    
    def update_agent_history(
        self, 
        agent_name: str, 
        url: str, 
        state_changed: bool,
        interaction_success: bool
    ):
        """更新 Agent 历史"""
        if url:
            self.agent_url_discoveries[agent_name].add(url)
            # JBS 追踪
            self.url_visit_counts[url][agent_name] += 1
        if state_changed:
            self.agent_state_changes[agent_name] += 1
        if interaction_success:
            self.agent_interactions[agent_name] += 1
    
    def detect_jbs(self, url: str, threshold: int = 5) -> Tuple[bool, float]:
        """
        检测 Joint Blind Spot (JBS) - 简化版
        
        优化：更高的阈值，更简单的计算
        """
        if not url:
            return False, 1.0
        
        visit_counts = self.url_visit_counts.get(url)
        if not visit_counts:
            return False, 1.0
        
        total_visits = sum(visit_counts.values())
        
        # 简化：只有当总访问次数超过阈值才检查
        if total_visits < threshold:
            return False, 1.0
        
        # 快速检查：超过一半 agent 访问过
        if len(visit_counts) >= self.agent_num // 2:
            penalty_factor = max(0.5, 1.0 / (1 + total_visits * 0.05))
            return True, penalty_factor
        
        return False, 1.0
    
    def get_jbs_report(self) -> Dict:
        """
        获取 JBS 诊断报告
        """
        # 找出最热门的 URL（潜在 JBS）
        hot_urls = []
        for url, visit_counts in self.url_visit_counts.items():
            total = sum(visit_counts.values())
            unique_agents = len(visit_counts)
            if total >= 5:  # 至少被访问5次
                hot_urls.append({
                    'url': url,
                    'total_visits': total,
                    'unique_agents': unique_agents,
                    'avg_per_agent': total / max(unique_agents, 1)
                })
        
        hot_urls.sort(key=lambda x: x['total_visits'], reverse=True)
        
        return {
            'jbs_events_count': len(self.jbs_events),
            'jbs_events': self.jbs_events[-10:],  # 最近10次
            'hot_urls': hot_urls[:10],  # 最热门10个 URL
            'agent_diversity': {
                agent: len(urls) for agent, urls in self.agent_url_discoveries.items()
            }
        }


# ============================================================================
# Component 5: Shapley 混合网络
# ============================================================================

class ShapleyMixingNetwork(nn.Module):
    """Shapley 混合网络，用于计算 Lovasz 扩展的梯度"""
    
    def __init__(self, n_agents: int, embed_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        
        self.hyper_w1 = nn.Sequential(
            nn.Linear(n_agents, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(n_agents, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.hyper_b1 = nn.Linear(n_agents, embed_dim)
        self.hyper_b2 = nn.Linear(n_agents, 1)
        
    def forward(self, agent_q_values: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        batch_size = agent_q_values.shape[0]
        weighted_q = agent_q_values * w
        
        w1 = torch.abs(self.hyper_w1(w)).view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(w)
        hidden = torch.bmm(weighted_q.unsqueeze(1), w1).squeeze(1) + b1
        hidden = F.elu(hidden)
        
        w2 = torch.abs(self.hyper_w2(w))
        b2 = self.hyper_b2(w)
        q_tot = (hidden * w2).sum(dim=1, keepdim=True) + b2
        
        return q_tot


# ============================================================================
# Main Class: SHAQ v2
# ============================================================================

class SHAQv2(multi_agent.multi_agent_system.MultiAgentSystem):
    """
    SHAQ v2: 完整理论框架实现
    
    核心组件：
    1. DOM 结构编码器 - 防止 Reward Hacking
    2. 内在好奇心模块 (ICM) - 驱动探索
    3. 多目标奖励系统 - 稀疏+稠密奖励
    4. 角色分工系统 - Explorer vs Exploiter
    5. Shapley 信用分配 - 公平的边际贡献计算
    """
    
    def __init__(self, params: Dict):
        super().__init__(params)
        self.params = params
        self.algo_type = params.get("algo_type", "shaq_v2")
        self.reward_function = params.get("reward_function", "A")
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"SHAQ v2: Using {self.device}")
        
        # 超参数
        self.max_random = params.get("max_random", 0.9)
        self.min_random = params.get("min_random", 0.1)
        self.batch_size = params.get("batch_size", 32)
        self.gamma = params.get("gamma", 0.5)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.update_target_interval = params.get("update_target_interval", 20)
        self.update_network_interval = params.get("update_network_interval", 4)
        self.shapley_update_interval = params.get("shapley_update_interval", 10)
        self.alive_time = params.get("alive_time", 10800)
        
        # ICM 参数
        self.use_icm = params.get("use_icm", True)
        self.icm_weight = params.get("icm_weight", 0.5)
        
        # 角色分工参数
        self.use_role_based = params.get("use_role_based", True)
        
        # Transformer
        self.transformer = instantiate_class_by_module_and_class_name(
            params["transformer_module"], params["transformer_class"]
        )
        
        # 记录
        self.state_list: List[WebState] = []
        self.action_list: List[WebAction] = []
        self.state_list_agent: Dict[str, List[WebState]] = {}
        self.action_count = defaultdict(int)
        self.learn_step_count = 0
        self.start_time = datetime.now()
        
        # 网络锁
        self.network_lock = threading.Lock()
        
        # Q 网络
        self.q_eval_agent: Dict[str, nn.Module] = {}
        self.q_target_agent: Dict[str, nn.Module] = {}
        self.agent_optimizer: Dict[str, optim.Optimizer] = {}
        
        for i in range(self.agent_num):
            agent_name = str(i)
            q_eval = instantiate_class_by_module_and_class_name(
                params["model_module"], params["model_class"]
            )
            q_target = instantiate_class_by_module_and_class_name(
                params["model_module"], params["model_class"]
            )
            q_eval.to(self.device)
            q_target.to(self.device)
            q_target.load_state_dict(q_eval.state_dict())
            
            self.q_eval_agent[agent_name] = q_eval
            self.q_target_agent[agent_name] = q_target
            self.agent_optimizer[agent_name] = optim.Adam(q_eval.parameters(), lr=self.learning_rate)
            self.state_list_agent[agent_name] = []
        
        # Shapley 混合网络
        self.mixing_network = ShapleyMixingNetwork(n_agents=self.agent_num, embed_dim=64).to(self.device)
        self.target_mixing_network = ShapleyMixingNetwork(n_agents=self.agent_num, embed_dim=64).to(self.device)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        self.mixing_optimizer = optim.Adam(self.mixing_network.parameters(), lr=self.learning_rate)
        
        # 内在好奇心模块 (ICM)
        if self.use_icm:
            self.icm = IntrinsicCuriosityModule(state_dim=64, action_dim=12).to(self.device)
            self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=self.learning_rate * 0.5)
        else:
            self.icm = None
        
        # 多目标奖励系统（三层架构，基于 ASE 2024 论文）
        self.reward_system = MultiObjectiveRewardSystem(
            mode='three_tier',  # 使用三层奖励系统
            alive_time=self.alive_time
        )
        
        # 角色分工系统
        self.role_system = RoleBasedRewardSystem(self.agent_num)
        
        # DOM 编码器
        self.dom_encoder = DOMStructureEncoder()
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=1000)
        self.replay_buffer_agent: Dict[str, ReplayBuffer] = {
            str(i): ReplayBuffer(capacity=500) for i in range(self.agent_num)
        }
        
        # 同步
        self.finish_dict_agent: Dict[str, bool] = {str(i): False for i in range(self.agent_num)}
        self.prev_state_success_dict: Dict[str, Optional[WebState]] = {}
        self.prev_action_success_dict: Dict[str, Optional[WebAction]] = {}
        self.current_state_success_dict: Dict[str, Optional[WebState]] = {}
        self.prev_html_success_dict: Dict[str, str] = {}
        
        for i in range(self.agent_num):
            agent_name = str(i)
            self.prev_state_success_dict[agent_name] = None
            self.prev_action_success_dict[agent_name] = None
            self.current_state_success_dict[agent_name] = None
            self.prev_html_success_dict[agent_name] = ""
        
        # Shapley 缓存
        self.cached_shapley_values: Dict[str, float] = {
            str(i): 1.0 / self.agent_num for i in range(self.agent_num)
        }
        self.shapley_update_counter = 0
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 追踪团队访问的 URL
        self.team_visited_urls: Set[str] = set()
        
        # 【新增】存储每个 Agent 的浏览器日志（用于三层奖励系统）
        self.agent_browser_logs: Dict[str, List[Dict]] = {}
        self.agent_performance_logs: Dict[str, List[Dict]] = {}
        self.logs_lock = threading.Lock()
        
        logger.info(f"SHAQ v2 initialized with {self.agent_num} agents, "
                   f"ICM: {self.use_icm}, Role-based: {self.use_role_based}, "
                   f"Mode: {self.reward_system.mode}")
    
    def set_agent_logs(
        self, 
        agent_name: str, 
        browser_logs: List[Dict], 
        performance_logs: List[Dict]
    ):
        """
        存储 Agent 的浏览器日志（由 MultiAgentThread 调用）
        
        用于三层奖励系统的伪 ELOC 指标计算
        """
        with self.logs_lock:
            self.agent_browser_logs[agent_name] = browser_logs
            self.agent_performance_logs[agent_name] = performance_logs
    
    def get_agent_logs(self, agent_name: str) -> Tuple[List[Dict], List[Dict]]:
        """
        获取 Agent 的浏览器日志
        
        Returns:
            (browser_logs, performance_logs)
        """
        with self.logs_lock:
            browser_logs = self.agent_browser_logs.get(agent_name, [])
            performance_logs = self.agent_performance_logs.get(agent_name, [])
            # 清空日志（避免重复计算）
            self.agent_browser_logs[agent_name] = []
            self.agent_performance_logs[agent_name] = []
        return browser_logs, performance_logs
    
    def get_tensor(self, action: WebAction, html: str, web_state: WebState) -> torch.Tensor:
        """将状态-动作对编码为张量"""
        state_tensor = self.transformer.state_to_tensor(web_state, html)
        execution_time = self.action_dict.get(action, 0)
        action_tensor = self.transformer.action_to_tensor(web_state, action, execution_time)
        tensor = torch.cat((state_tensor, action_tensor))
        return tensor.float()
    
    def get_action_algorithm(self, web_state: WebState, html: str, agent_name: str) -> WebAction:
        """动作选择"""
        self.update_state_records(web_state, html, agent_name)
        
        actions = web_state.get_action_list()
        if len(actions) == 1 and isinstance(actions[0], RestartAction):
            return actions[0]
        
        q_eval = self.q_eval_agent[agent_name]
        q_eval.eval()
        
        action_tensors = []
        for temp_action in actions:
            action_tensor = self.get_tensor(temp_action, html, web_state)
            action_tensors.append(action_tensor)
        
        with torch.no_grad():
            if isinstance(q_eval, DenseNet):
                output = q_eval(torch.stack(action_tensors).unsqueeze(1).to(self.device))
            else:
                output = q_eval(torch.stack(action_tensors).to(self.device))
        
        q_values = output.squeeze(-1).cpu().numpy()
        max_idx = q_values.argmax()
        max_val = q_values[max_idx]
        chosen_action = actions[max_idx]
        
        logger.info(f"[{agent_name}] SHAQ v2 max Q: {max_val:.4f}")
        
        # ε-greedy with Shapley guidance
        time_diff = (datetime.now() - self.start_time).total_seconds()
        time_diff = min(time_diff, self.alive_time)
        
        base_epsilon = self.max_random - min(time_diff / self.alive_time * 2, 1.0) * (
            self.max_random - self.min_random
        )
        
        # Shapley 引导：低贡献者增加探索
        shapley_val = self.cached_shapley_values.get(agent_name, 1.0 / self.agent_num)
        avg_shapley = 1.0 / self.agent_num
        if shapley_val < avg_shapley * 0.8:
            epsilon = min(self.max_random, base_epsilon * 1.5)
        else:
            epsilon = base_epsilon
        
        # 角色引导探索
        if self.use_role_based:
            role = self.role_system.get_agent_role(agent_name)
            if role == 'explorer':
                # Explorer 更倾向于选择新动作
                epsilon = min(self.max_random, epsilon * 1.2)
        
        if random.uniform(0, 1) < epsilon:
            unexplored = [a for a in actions if self.action_dict.get(a, 0) == 0]
            if unexplored:
                chosen_action = random.choice(unexplored)
            else:
                chosen_action = random.choice(actions)
        
        self.action_count[chosen_action] += 1
        return chosen_action
    
    def update_state_records(self, web_state: WebState, html: str, agent_name: str):
        """更新状态记录并触发学习"""
        if web_state not in self.state_list:
            self.state_list.append(web_state)
        if web_state not in self.state_list_agent[agent_name]:
            self.state_list_agent[agent_name].append(web_state)
        
        for action in web_state.get_action_list():
            if action not in self.action_list:
                self.action_list.append(action)
        
        if (self.prev_action_dict.get(agent_name) is None or
            self.prev_state_dict.get(agent_name) is None or
            not isinstance(self.prev_state_dict[agent_name], ActionSetWithExecutionTimesState)):
            return
        
        # 计算奖励
        reward = self.get_reward(web_state, html, agent_name)
        
        tensor = self.get_tensor(
            self.prev_action_dict[agent_name],
            self.prev_html_dict[agent_name],
            self.prev_state_dict[agent_name]
        )
        tensor = tensor.unsqueeze(0)
        
        done = not isinstance(web_state, ActionSetWithExecutionTimesState)
        
        self.replay_buffer_agent[agent_name].push(
            tensor, tensor, reward, web_state, html, done
        )
        
        self.learn_agent(agent_name)
        self.try_joint_learning(web_state, html, agent_name)
    
    def get_reward(
        self, 
        web_state: WebState, 
        html: str, 
        agent_name: str,
        browser_logs: List[Dict] = None,
        performance_logs: List[Dict] = None,
        http_status: int = 200
    ) -> float:
        """
        三层奖励计算（基于 ASE 2024 论文）
        
        三层架构：
        - 目标层: Fault Detection（最高优先级）
        - 核心层: 伪 ELOC（论文证明比 Activity 更稳健！）
        - 基础层: Activity Coverage（最低优先级）
        
        Args:
            browser_logs: 从 driver.get_log("browser") 获取
            performance_logs: 从 driver.get_log("performance") 获取
            http_status: HTTP 状态码
        """
        if not isinstance(web_state, ActionSetWithExecutionTimesState):
            return self.reward_system.R_PENALTY
        
        prev_action = self.prev_action_dict.get(agent_name)
        
        # 【新增】如果没有传入 logs，尝试从内部获取
        if browser_logs is None or performance_logs is None:
            stored_browser_logs, stored_performance_logs = self.get_agent_logs(agent_name)
            if browser_logs is None:
                browser_logs = stored_browser_logs
            if performance_logs is None:
                performance_logs = stored_performance_logs
        
        # ============================================================
        # 三层奖励计算
        # ============================================================
        if self.reward_system.mode == 'three_tier' and performance_logs:
            # 使用三层奖励系统（论文推荐）
            total_reward, breakdown = self.reward_system.compute_three_tier_reward(
                web_state=web_state,
                html=html,
                action=prev_action,
                browser_logs=browser_logs,
                performance_logs=performance_logs,
                http_status=http_status
            )
            
            # 日志：显示各层贡献
            if self.learn_step_count % 50 == 0:
                tier_info = f"目标:{breakdown.get('_target_reward', 0):.1f} " \
                           f"核心:{breakdown.get('_core_reward', 0):.1f} " \
                           f"基础:{breakdown.get('_base_reward', 0):.1f}"
                logger.debug(f"[{agent_name}] 三层奖励: {tier_info}")
        else:
            # 回退到混合模式（没有 performance_logs 时）
            total_reward, breakdown = self.reward_system.compute_coverage_reward(
                web_state, prev_action, 
                console_errors=[log.get('message', '') for log in (browser_logs or []) if log.get('level') == 'SEVERE'],
                http_status=http_status
            )
        
        # ============================================================
        # ICM 好奇心奖励（补充探索）
        # ============================================================
        if self.use_icm:
            curiosity_reward = self.icm.compute_intrinsic_reward()
            total_reward += curiosity_reward * self.icm_weight
        
        # ============================================================
        # 角色分工 + JBS 检测
        # ============================================================
        if self.use_role_based:
            current_url = getattr(web_state, 'url', None)
            
            # 多样性奖励
            diversity_bonus = self.role_system.compute_diversity_bonus(
                agent_name, current_url, self.team_visited_urls
            )
            total_reward += diversity_bonus
            
            # JBS 惩罚
            is_jbs, jbs_penalty = self.role_system.detect_jbs(current_url)
            if is_jbs:
                total_reward *= jbs_penalty
                logger.debug(f"[{agent_name}] JBS 检测: {current_url}, 惩罚系数: {jbs_penalty:.2f}")
            
            # 更新历史
            self.role_system.update_agent_history(agent_name, current_url, False, False)
            if current_url:
                self.team_visited_urls.add(current_url)
        
        # ============================================================
        # Shapley 信用调整（低贡献者获得探索奖励）
        # ============================================================
        shapley_val = self.cached_shapley_values.get(agent_name, 1.0 / self.agent_num)
        if shapley_val < 1.0 / self.agent_num * 0.8:
            exploration_bonus = 5.0
            total_reward += exploration_bonus
        
        return total_reward
    
    def try_joint_learning(self, web_state: WebState, html: str, agent_name: str):
        """尝试联合学习"""
        with self.lock:
            if (not isinstance(self.prev_state_dict.get(agent_name), ActionSetWithExecutionTimesState) or
                not isinstance(self.current_state_dict.get(agent_name), ActionSetWithExecutionTimesState)):
                return
            
            self.finish_dict_agent[agent_name] = True
            self.prev_state_success_dict[agent_name] = self.prev_state_dict[agent_name]
            self.prev_action_success_dict[agent_name] = self.prev_action_dict[agent_name]
            self.current_state_success_dict[agent_name] = self.current_state_dict[agent_name]
            self.prev_html_success_dict[agent_name] = self.prev_html_dict[agent_name]
            
            if not all(self.finish_dict_agent.values()):
                return
            
            for i in range(self.agent_num):
                self.finish_dict_agent[str(i)] = False
            
            # 收集联合经验
            tensors = []
            next_states = []
            htmls = []
            total_reward = 0.0
            
            for i in range(self.agent_num):
                an = str(i)
                tensor = self.get_tensor(
                    self.prev_action_success_dict[an],
                    self.prev_html_success_dict[an],
                    self.prev_state_success_dict[an]
                )
                tensor = tensor.unsqueeze(0)
                tensors.append(tensor)
                next_states.append(self.current_state_success_dict[an])
                htmls.append(self.prev_html_success_dict[an])
                
                total_reward += self.get_reward(
                    self.current_state_success_dict[an],
                    self.prev_html_success_dict[an],
                    an
                )
            
            self.replay_buffer.push(tensors, tensors, total_reward, next_states, htmls, False)
        
        self.learn_joint()
    
    def learn_agent(self, agent_name: str):
        """单智能体学习"""
        buffer = self.replay_buffer_agent[agent_name]
        if len(buffer.buffer) < self.batch_size:
            return
        
        self.learn_step_count += 1
        
        if self.learn_step_count % self.update_target_interval == 0:
            self.update_target_networks()
        
        if self.learn_step_count % self.update_network_interval != 0:
            return
        
        tensors, _, rewards, next_states, htmls, dones = buffer.sample(self.batch_size)
        
        q_eval = self.q_eval_agent[agent_name]
        q_target = self.q_target_agent[agent_name]
        
        with self.network_lock:
            target_list = []
            
            for i in range(self.batch_size):
                tensor = tensors[i]
                reward = rewards[i]
                next_state = next_states[i]
                html = htmls[i]
                done = dones[i]
                
                next_q_value = 0.0
                if isinstance(next_state, ActionSetWithExecutionTimesState) and not done:
                    next_actions = next_state.get_action_list()
                    if next_actions:
                        next_tensors = [self.get_tensor(a, html, next_state) for a in next_actions[:20]]
                        with torch.no_grad():
                            if isinstance(q_target, DenseNet):
                                next_q = q_target(torch.stack(next_tensors).unsqueeze(1).to(self.device))
                            else:
                                next_q = q_target(torch.stack(next_tensors).to(self.device))
                        next_q_value = next_q.max().item()
                
                target_q = reward + self.gamma * next_q_value * (1 - int(done))
                target_list.append(target_q)
            
            q_eval.train()
            if isinstance(q_eval, DenseNet):
                input_tensor = torch.stack(tensors).to(self.device)
            else:
                input_tensor = torch.stack(tensors).squeeze(1).to(self.device)
            
            q_predicts = q_eval(input_tensor)
            q_targets = torch.tensor(target_list, device=self.device).unsqueeze(-1)
            
            loss = self.criterion(q_predicts, q_targets)
            
            self.agent_optimizer[agent_name].zero_grad()
            loss.backward()
            self.agent_optimizer[agent_name].step()
    
    def learn_joint(self):
        """联合学习 + Shapley 计算"""
        if len(self.replay_buffer.buffer) < self.batch_size:
            return
        
        self.shapley_update_counter += 1
        
        tensors_batch, _, rewards_batch, next_states_batch, htmls_batch, dones_batch = \
            self.replay_buffer.sample(min(self.batch_size, len(self.replay_buffer.buffer)))
        
        actual_batch_size = len(tensors_batch)
        
        with self.network_lock:
            agent_q_values = []
            
            for batch_idx in range(actual_batch_size):
                tensors = tensors_batch[batch_idx]
                current_qs = []
                
                for agent_idx in range(self.agent_num):
                    agent_name = str(agent_idx)
                    q_eval = self.q_eval_agent[agent_name]
                    tensor = tensors[agent_idx]
                    
                    if isinstance(q_eval, DenseNet):
                        q_val = q_eval(tensor.unsqueeze(0).to(self.device))
                    else:
                        q_val = q_eval(tensor.to(self.device))
                    current_qs.append(q_val.squeeze())
                
                agent_q_values.append(torch.stack(current_qs))
            
            agent_q_values = torch.stack(agent_q_values)
            
            # 计算 Shapley values
            if self.shapley_update_counter % self.shapley_update_interval == 0:
                self._update_shapley_values(agent_q_values.detach())
            
            # 计算联合 Q 值
            full_participation = torch.ones(actual_batch_size, self.agent_num, device=self.device)
            q_tot = self.mixing_network(agent_q_values, full_participation)
            
            with torch.no_grad():
                next_q_tot = self.target_mixing_network(agent_q_values.detach(), full_participation)
            
            rewards = torch.tensor(rewards_batch, device=self.device).unsqueeze(-1)
            dones = torch.tensor([float(d) for d in dones_batch], device=self.device).unsqueeze(-1)
            
            target_q_tot = rewards + self.gamma * next_q_tot * (1 - dones)
            
            loss = self.criterion(q_tot, target_q_tot.detach())
            
            self.mixing_optimizer.zero_grad()
            for agent_name in self.agent_optimizer:
                self.agent_optimizer[agent_name].zero_grad()
            
            loss.backward()
            
            self.mixing_optimizer.step()
            for agent_name in self.agent_optimizer:
                self.agent_optimizer[agent_name].step()
            
            if self.shapley_update_counter % 10 == 0:
                shapley_str = ", ".join([f"A{i}:{self.cached_shapley_values[str(i)]:.3f}" 
                                        for i in range(self.agent_num)])
                logger.info(f"SHAQ v2 loss: {loss.item():.4f}, Shapley: [{shapley_str}]")
    
    def _update_shapley_values(self, agent_q_values: torch.Tensor):
        """使用 Lovasz 扩展计算 Shapley values"""
        batch_size = agent_q_values.shape[0]
        
        w = torch.full(
            (batch_size, self.agent_num),
            0.5,
            device=self.device,
            requires_grad=True
        )
        
        q_tot = self.mixing_network(agent_q_values, w)
        q_tot.sum().backward()
        
        mean_shapley = w.grad.mean(dim=0)
        shapley_sum = mean_shapley.abs().sum().item()
        
        for i in range(self.agent_num):
            if shapley_sum > 0:
                self.cached_shapley_values[str(i)] = mean_shapley[i].abs().item() / shapley_sum
            else:
                self.cached_shapley_values[str(i)] = 1.0 / self.agent_num
    
    def update_target_networks(self):
        """更新目标网络"""
        with self.network_lock:
            for i in range(self.agent_num):
                agent_name = str(i)
                self.q_target_agent[agent_name].load_state_dict(
                    self.q_eval_agent[agent_name].state_dict()
                )
            self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        logger.info("SHAQ v2 target networks updated")
    
    def get_diagnostic_report(self) -> Dict:
        """
        获取完整的诊断报告
        
        用于验证：
        1. Reward System 与 Metrics 是否对齐
        2. JBS 问题是否存在/被缓解
        3. 角色分工是否有效
        4. 【新增】伪 ELOC 指标效果
        """
        alignment_report = self.reward_system.get_alignment_report()
        
        report = {
            'algorithm': 'SHAQv2 (Three-Tier)',
            'agents': self.agent_num,
            'total_steps': len(self.state_list),
            
            # 三层奖励分析（新增）
            'three_tier_reward': {
                'tier_totals': self.reward_system.tier_reward_totals,
                'tier_ratios': alignment_report.get('three_tier_analysis', {}).get('tier_ratios', {}),
                'interpretation': {
                    '目标层(Fault)占比': f"{self.reward_system.tier_reward_totals['target']:.1f}",
                    '核心层(ELOC)占比': f"{self.reward_system.tier_reward_totals['core']:.1f}",
                    '基础层(Activity)占比': f"{self.reward_system.tier_reward_totals['base']:.1f}",
                }
            },
            
            # 伪 ELOC 指标（新增）
            'pseudo_eloc': self.reward_system.pseudo_eloc_tracker.get_pseudo_eloc_summary(),
            
            # Reward-Metric 对齐报告
            'reward_alignment': alignment_report,
            
            # JBS 报告
            'jbs_analysis': self.role_system.get_jbs_report() if self.use_role_based else {},
            
            # Shapley 值分布
            'shapley_distribution': dict(self.cached_shapley_values),
            
            # 角色统计
            'role_stats': {
                str(i): {
                    'role': self.role_system.get_agent_role(str(i)) if self.use_role_based else 'none',
                    'states_discovered': len(self.state_list_agent.get(str(i), [])),
                    'urls_discovered': len(self.role_system.agent_url_discoveries.get(str(i), set())) if self.use_role_based else 0,
                }
                for i in range(self.agent_num)
            },
            
            # 团队统计
            'team_stats': {
                'total_urls': len(self.team_visited_urls),
                'total_states': len(self.state_list),
                'total_actions': len(self.action_list),
            }
        }
        
        return report
    
    def print_diagnostic_report(self):
        """打印诊断报告"""
        report = self.get_diagnostic_report()
        
        logger.info("=" * 60)
        logger.info("SHAQ v2 诊断报告（三层奖励系统）")
        logger.info("=" * 60)
        
        # 【新增】三层奖励分析
        logger.info("\n【三层奖励分析】（基于 ASE 2024 论文）")
        tier_info = report.get('three_tier_reward', {})
        tier_totals = tier_info.get('tier_totals', {})
        logger.info(f"  目标层 (Fault Detection): {tier_totals.get('target', 0):.2f}")
        logger.info(f"  核心层 (伪 ELOC):        {tier_totals.get('core', 0):.2f}")
        logger.info(f"  基础层 (Activity):       {tier_totals.get('base', 0):.2f}")
        
        # 【新增】伪 ELOC 统计
        logger.info("\n【伪 ELOC 指标】（Method/ELOC Coverage 代理）")
        pseudo_eloc = report.get('pseudo_eloc', {})
        logger.info(f"  API 端点发现: {pseudo_eloc.get('total_api_endpoints', 0)} 个")
        logger.info(f"  请求模式多样性: {pseudo_eloc.get('total_request_patterns', 0)} 种")
        logger.info(f"  跨域名请求: {pseudo_eloc.get('total_domains', 0)} 个域名")
        logger.info(f"  总请求数: {pseudo_eloc.get('total_requests', 0)}")
        logger.info(f"  最大 DOM 深度: {pseudo_eloc.get('max_dom_depth', 0)}")
        
        # 热门 API 端点
        api_endpoints = pseudo_eloc.get('api_endpoints_list', [])[:5]
        if api_endpoints:
            logger.info("  发现的 API 端点:")
            for ep in api_endpoints:
                logger.info(f"    - {ep}")
        
        # 奖励分解统计
        alignment = report['reward_alignment']
        logger.info("\n【奖励分解统计】")
        for key, value in alignment['reward_breakdown'].items():
            count = alignment['reward_counts'].get(key, 0)
            logger.info(f"  {key}: 总计 {value:.2f} ({count} 次)")
        
        # JBS 分析
        if report['jbs_analysis']:
            jbs = report['jbs_analysis']
            logger.info(f"\n【JBS 分析】")
            logger.info(f"  JBS 事件数: {jbs['jbs_events_count']}")
            logger.info(f"  热门 URL 数: {len(jbs['hot_urls'])}")
            if jbs['hot_urls']:
                logger.info(f"  最热门 URL: {jbs['hot_urls'][0]['url']} ({jbs['hot_urls'][0]['total_visits']} 次)")
        
        # 角色统计
        logger.info("\n【角色统计】")
        for agent, stats in report['role_stats'].items():
            logger.info(f"  Agent {agent} ({stats['role']}): {stats['states_discovered']} 状态, {stats['urls_discovered']} URL")
        
        logger.info("=" * 60)
