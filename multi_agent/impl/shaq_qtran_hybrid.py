"""
SHAQ-QTRAN Hybrid: 结合两种算法的优势

设计思路：
1. 动作选择：使用 QTRAN 的快速 Q 网络（速度快）
2. 值分解：使用 QTRAN 的 QTranNetwork（高效）
3. 信用分配：使用 SHAQ 的 Shapley value（公平）
4. 探索策略：结合两者优点

核心创新：
- 用 Shapley value 调整个体奖励，但不影响动作选择速度
- Shapley 计算频率低（每 N 步一次），减少开销
- 保持 QTRAN 的速度优势，获得 SHAQ 的公平性
"""

import math
import random
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import multi_agent.multi_agent_system
from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from model.dense_net import DenseNet
from model.mixing_network import QTranNetwork
from model.replay_buffer import ReplayBuffer
from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState
from state.web_state import WebState
from utils import instantiate_class_by_module_and_class_name
from web_test.multi_agent_thread import logger


class ShapleyMixingNetwork(nn.Module):
    """轻量级 Shapley 混合网络，用于计算 Shapley value"""
    
    def __init__(self, n_agents: int, embed_dim: int = 32):
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


class SHAQQTRANHybrid(multi_agent.multi_agent_system.MultiAgentSystem):
    """
    SHAQ-QTRAN 混合算法
    
    核心特性：
    1. 使用 QTRAN 的动作选择和值分解（速度）
    2. 使用 SHAQ 的 Shapley value 进行信用分配（公平）
    3. 低频率 Shapley 计算，不影响整体性能
    """
    
    def __init__(self, params: Dict):
        super().__init__(params)
        self.params = params
        self.algo_type = params.get("algo_type", "shaq_qtran")
        self.reward_function = params.get("reward_function", "A")
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"SHAQ-QTRAN Hybrid: Using {self.device}")
        
        # 基础参数（与 QTRAN 对齐）
        self.max_random = params.get("max_random", 0.9)
        self.min_random = params.get("min_random", 0.3)
        self.batch_size = params.get("batch_size", 32)
        self.mix_batch_size = params.get("mix_batch_size", 16)
        self.gamma = params.get("gamma", 0.5)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.update_target_interval = params.get("update_target_interval", 8)
        self.update_network_interval = params.get("update_network_interval", 4)
        self.update_mixing_network_interval = params.get("update_mixing_network_interval", 1)
        self.alive_time = params.get("alive_time", 10800)
        
        # SHAQ 特有参数（低频率计算）
        self.shapley_update_interval = params.get("shapley_update_interval", 20)
        self.shapley_weight = params.get("shapley_weight", 0.3)  # Shapley 在奖励中的权重
        
        # Transformer
        self.transformer = instantiate_class_by_module_and_class_name(
            params["transformer_module"], params["transformer_class"]
        )
        
        # 记录数据
        self.state_list: List[WebState] = []
        self.action_list: List[WebAction] = []
        self.action_count = defaultdict(int)
        self.state_list_agent: Dict[str, List[WebState]] = {}
        self.learn_step_count = 0
        self.start_time = datetime.now()
        
        # 网络锁
        self.network_lock = threading.Lock()
        
        # 各智能体的 Q 网络
        self.q_eval_agent: Dict[str, nn.Module] = {}
        self.q_target_agent: Dict[str, nn.Module] = {}
        self.agent_optimizer: Dict[str, optim.Optimizer] = {}
        
        # 初始化智能体网络
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
        
        # QTRAN 混合网络（用于值分解）
        self.mixing_network = QTranNetwork(
            n_agents=self.agent_num, 
            state_dim=self.agent_num * 52, 
            action_dim=12
        ).to(self.device)
        self.target_mixing_network = QTranNetwork(
            n_agents=self.agent_num,
            state_dim=self.agent_num * 52,
            action_dim=12
        ).to(self.device)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        self.mixing_optimizer = optim.Adam(self.mixing_network.parameters(), lr=self.learning_rate)
        
        # SHAQ Shapley 网络（轻量级，用于信用分配）
        self.shapley_network = ShapleyMixingNetwork(n_agents=self.agent_num, embed_dim=32).to(self.device)
        
        # 经验回放
        self.replay_buffer_mixing = ReplayBuffer(capacity=500)
        self.replay_buffer_agent: Dict[str, ReplayBuffer] = {
            str(i): ReplayBuffer(capacity=500) for i in range(self.agent_num)
        }
        
        # 同步相关
        self.finish_dict_agent: Dict[str, bool] = {str(i): False for i in range(self.agent_num)}
        self.prev_state_success_dict: Dict[str, Optional[WebState]] = {}
        self.prev_action_success_dict: Dict[str, Optional[WebAction]] = {}
        self.current_state_success_dict: Dict[str, Optional[WebState]] = {}
        self.prev_html_success_dict: Dict[str, str] = {}
        self.prev_best_action_dict: Dict[str, Optional[WebAction]] = {}
        self.prev_best_action_success_dict: Dict[str, Optional[WebAction]] = {}
        
        for i in range(self.agent_num):
            agent_name = str(i)
            self.prev_state_success_dict[agent_name] = None
            self.prev_action_success_dict[agent_name] = None
            self.current_state_success_dict[agent_name] = None
            self.prev_html_success_dict[agent_name] = ""
            self.prev_best_action_dict[agent_name] = None
            self.prev_best_action_success_dict[agent_name] = None
        
        # Shapley 缓存
        self.cached_shapley_values: Dict[str, float] = {
            str(i): 1.0 / self.agent_num for i in range(self.agent_num)
        }
        self.shapley_update_counter = 0
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 奖励常量
        self.R_PENALTY = -99.0
        self.R_A_PENALTY = -5.0
        self.R_A_BASE_HIGH = 50.0
        self.R_A_BASE_MIDDLE = 10.0
        self.R_A_MIN_SIM_LINE = 0.7
        self.R_A_MIDDLE_SIM_LINE = 0.85
        
        logger.info(f"SHAQ-QTRAN Hybrid initialized with {self.agent_num} agents, "
                   f"shapley_interval: {self.shapley_update_interval}, "
                   f"shapley_weight: {self.shapley_weight}")
    
    def get_tensor(self, action: WebAction, html: str, web_state: WebState) -> torch.Tensor:
        """将状态-动作对编码为张量"""
        state_tensor = self.transformer.state_to_tensor(web_state, html)
        execution_time = self.action_dict.get(action, 0)
        action_tensor = self.transformer.action_to_tensor(web_state, action, execution_time)
        tensor = torch.cat((state_tensor, action_tensor))
        return tensor.float()
    
    def get_action_algorithm(self, web_state: WebState, html: str, agent_name: str) -> WebAction:
        """
        动作选择 - 使用 QTRAN 风格的快速选择
        """
        self.update(web_state, html, agent_name)
        
        actions = web_state.get_action_list()
        if len(actions) == 1 and isinstance(actions[0], RestartAction):
            return actions[0]
        
        # 使用 Q 网络选择动作（QTRAN 风格）
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
        self.prev_best_action_dict[agent_name] = chosen_action
        
        logger.info(f"[{agent_name}] Hybrid max Q-value: {max_val:.4f}")
        
        # ε-greedy 探索（QTRAN 风格）
        end_time = datetime.now()
        time_diff = (end_time - self.start_time).total_seconds()
        time_diff = min(time_diff, self.alive_time)
        
        epsilon = self.max_random - min(time_diff / self.alive_time * 2, 1.0) * (
            self.max_random - self.min_random
        )
        
        # Shapley 引导的探索调整
        shapley_val = self.cached_shapley_values.get(agent_name, 1.0 / self.agent_num)
        avg_shapley = 1.0 / self.agent_num
        if shapley_val < avg_shapley * 0.8:  # 低贡献者增加探索
            epsilon = min(self.max_random, epsilon * 1.3)
        
        if random.uniform(0, 1) < epsilon:
            # 优先选择未执行过的动作
            unexplored = [a for a in actions if self.action_dict.get(a, 0) == 0]
            if unexplored:
                chosen_action = random.choice(unexplored)
            else:
                chosen_action = random.choice(actions)
        
        self.action_count[chosen_action] += 1
        return chosen_action
    
    def get_reward(self, web_state: WebState, agent_name: str) -> float:
        """
        计算奖励 - 结合基础奖励和 Shapley 信用
        """
        if self.reward_function != "A":
            return 0.0
        
        if not isinstance(web_state, ActionSetWithExecutionTimesState):
            return self.R_A_PENALTY
        
        # 计算状态新颖度
        max_sim = -1.0
        for temp_state in self.state_list[-200:]:
            if isinstance(temp_state, (OutOfDomainState, ActionExecuteFailedState, SameUrlState)):
                continue
            if web_state == temp_state:
                continue
            sim = web_state.similarity(temp_state)
            if sim > max_sim:
                max_sim = sim
        
        # 基础状态奖励
        if max_sim < self.R_A_MIN_SIM_LINE:
            r_state = self.R_A_BASE_HIGH
        elif max_sim < self.R_A_MIDDLE_SIM_LINE:
            r_state = self.R_A_BASE_MIDDLE
        else:
            visited_time = self.state_dict.get(web_state, 0)
            r_state = 2.0 if visited_time == 0 else 1.0 / visited_time
        
        # 动作奖励
        prev_action = self.prev_action_dict.get(agent_name)
        if prev_action and prev_action in self.action_list:
            exec_count = self.action_count.get(prev_action, 0)
            r_action = 2.0 if exec_count == 0 else 1.0 / float(max(1, exec_count))
        else:
            r_action = 0.0
        
        # 时间因子
        time_diff = (datetime.now() - self.start_time).total_seconds()
        r_time = 1 + float(time_diff) / self.alive_time
        
        # 基础奖励
        base_reward = (r_state + r_action) * r_time
        
        # Shapley 信用调整
        shapley_val = self.cached_shapley_values.get(agent_name, 1.0 / self.agent_num)
        shapley_factor = max(0.8, min(1.2, shapley_val * self.agent_num))
        
        # 混合奖励：(1-w)*base + w*shapley_adjusted
        final_reward = (1 - self.shapley_weight) * base_reward + self.shapley_weight * base_reward * shapley_factor
        
        return final_reward
    
    def get_reward_total(self) -> float:
        """计算团队总奖励"""
        total = 0.0
        for i in range(self.agent_num):
            agent_name = str(i)
            state = self.current_state_success_dict.get(agent_name)
            if state is not None:
                total += self.get_reward(state, agent_name)
        return total
    
    def update(self, web_state: WebState, html: str, agent_name: str):
        """更新状态记录并触发学习"""
        # 更新状态列表
        if web_state not in self.state_list:
            self.state_list.append(web_state)
        if web_state not in self.state_list_agent[agent_name]:
            self.state_list_agent[agent_name].append(web_state)
        
        # 更新动作列表
        for action in web_state.get_action_list():
            if action not in self.action_list:
                self.action_list.append(action)
        
        if (self.prev_action_dict.get(agent_name) is None or
            self.prev_state_dict.get(agent_name) is None or
            not isinstance(self.prev_state_dict[agent_name], ActionSetWithExecutionTimesState)):
            return
        
        tensor = self.get_tensor(
            self.prev_action_dict[agent_name],
            self.prev_html_dict[agent_name],
            self.prev_state_dict[agent_name]
        )
        tensor.unsqueeze_(0)
        
        # 存储智能体经验并学习
        reward = self.get_reward(web_state, agent_name)
        self.replay_buffer_agent[agent_name].push(tensor, tensor, reward, web_state, html, False)
        self.learn_agent(agent_name)
        
        # 联合学习（QTRAN + Shapley）
        with self.lock:
            if (not isinstance(self.prev_state_dict.get(agent_name), ActionSetWithExecutionTimesState) or
                not isinstance(self.current_state_dict.get(agent_name), ActionSetWithExecutionTimesState)):
                return
            
            self.finish_dict_agent[agent_name] = True
            self.prev_state_success_dict[agent_name] = self.prev_state_dict[agent_name]
            self.prev_action_success_dict[agent_name] = self.prev_action_dict[agent_name]
            self.prev_best_action_success_dict[agent_name] = self.prev_best_action_dict[agent_name]
            self.current_state_success_dict[agent_name] = self.current_state_dict[agent_name]
            self.prev_html_success_dict[agent_name] = self.prev_html_dict[agent_name]
            
            if not all(self.finish_dict_agent.values()):
                return
            
            for i in range(self.agent_num):
                self.finish_dict_agent[str(i)] = False
            
            # 收集联合经验
            tensors = []
            best_tensors = []
            next_states = []
            htmls = []
            
            for i in range(self.agent_num):
                an = str(i)
                t = self.get_tensor(
                    self.prev_action_success_dict[an],
                    self.prev_html_success_dict[an],
                    self.prev_state_success_dict[an]
                )
                t.unsqueeze_(0)
                tensors.append(t)
                
                bt = self.get_tensor(
                    self.prev_best_action_success_dict[an],
                    self.prev_html_success_dict[an],
                    self.prev_state_success_dict[an]
                )
                bt.unsqueeze_(0)
                best_tensors.append(bt)
                
                next_states.append(self.current_state_success_dict[an])
                htmls.append(self.prev_html_success_dict[an])
            
            reward_total = self.get_reward_total()
            self.replay_buffer_mixing.push(tensors, best_tensors, reward_total, next_states, htmls, False)
        
        self.learn_mixing()
    
    def learn_agent(self, agent_name: str):
        """单智能体学习（QTRAN 风格）"""
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
    
    def learn_mixing(self):
        """联合学习：QTRAN 值分解 + Shapley 信用"""
        if len(self.replay_buffer_mixing.buffer) < self.mix_batch_size:
            return
        
        self.shapley_update_counter += 1
        
        tensors, best_tensors, rewards, next_states, htmls, dones = \
            self.replay_buffer_mixing.sample(self.mix_batch_size)
        
        batch_current_q_values = []
        batch_current_best_q_values = []
        batch_next_q_values = []
        batch_states = []
        batch_next_states = []
        batch_actions = []
        batch_best_actions = []
        batch_next_actions = []
        
        with self.network_lock:
            for i in range(self.mix_batch_size):
                tensor = tensors[i]
                best_tensor = best_tensors[i]
                next_state = next_states[i]
                html = htmls[i]
                
                current_q_values = []
                current_best_q_values = []
                next_q_values = []
                state_total = None
                next_state_total = None
                actions = []
                best_actions = []
                next_actions = []
                
                for j in range(self.agent_num):
                    agent_name = str(j)
                    q_eval = self.q_eval_agent[agent_name]
                    q_target = self.q_target_agent[agent_name]
                    
                    if isinstance(q_eval, DenseNet):
                        input_vector = tensor[j].unsqueeze(0)
                        best_vector = best_tensor[j].unsqueeze(0)
                    else:
                        input_vector = tensor[j]
                        best_vector = best_tensor[j]
                    
                    if state_total is None:
                        state_total = tensor[j].squeeze(0)
                    else:
                        state_total = torch.cat((state_total, tensor[j].squeeze(0)))
                    
                    output = q_eval(input_vector.to(self.device))
                    current_q_values.append(output.squeeze(0))
                    
                    output = q_eval(best_vector.to(self.device))
                    current_best_q_values.append(output.squeeze(0))
                    
                    next_q_value = torch.tensor(0.0, device=self.device)
                    best_action_tensor = torch.zeros_like(input_vector.squeeze(0).squeeze(0))
                    
                    if isinstance(next_state[j], ActionSetWithExecutionTimesState):
                        action_list = next_state[j].get_action_list()
                        action_tensors = [self.get_tensor(a, html[j], next_state[j]) for a in action_list[:20]]
                        if action_tensors:
                            if isinstance(q_eval, DenseNet):
                                q_values = q_target(torch.stack(action_tensors).unsqueeze(1).to(self.device)).squeeze(1)
                            else:
                                q_values = q_target(torch.stack(action_tensors).to(self.device)).squeeze(1)
                            next_q_value, max_index = q_values.max(0)
                            best_action_tensor = action_tensors[max_index.item()]
                    
                    next_q_values.append(next_q_value)
                    if next_state_total is None:
                        next_state_total = best_action_tensor
                    else:
                        next_state_total = torch.cat((next_state_total, best_action_tensor))
                    
                    actions.append(input_vector.squeeze(0).squeeze(0)[-12:])
                    best_actions.append(best_vector.squeeze(0).squeeze(0)[-12:])
                    next_actions.append(best_action_tensor[-12:])
                
                batch_current_q_values.append(torch.stack(current_q_values))
                batch_current_best_q_values.append(torch.stack(current_best_q_values))
                batch_next_q_values.append(torch.stack(next_q_values))
                batch_states.append(state_total)
                batch_next_states.append(next_state_total)
                batch_actions.append(torch.stack(actions))
                batch_best_actions.append(torch.stack(best_actions))
                batch_next_actions.append(torch.stack(next_actions))
            
            batch_current_q_values = torch.stack(batch_current_q_values).squeeze(-1)
            batch_current_best_q_values = torch.stack(batch_current_best_q_values).squeeze(-1)
            batch_next_q_values = torch.stack(batch_next_q_values).squeeze(-1)
            batch_states = torch.stack(batch_states)
            batch_next_states = torch.stack(batch_next_states)
            batch_actions = torch.stack(batch_actions)
            batch_best_actions = torch.stack(batch_best_actions)
            batch_next_actions = torch.stack(batch_next_actions)
            rewards_tensor = torch.tensor(rewards).view(-1, 1).to(self.device)
            
            # QTRAN 损失
            joint_q_value, v_value = self.mixing_network(
                batch_states.to(self.device), batch_actions.to(self.device)
            )
            best_joint_q_value, best_v_value = self.mixing_network(
                batch_states.to(self.device), batch_best_actions.to(self.device)
            )
            next_joint_q_value, next_v_value = self.target_mixing_network(
                batch_next_states.to(self.device), batch_next_actions.to(self.device)
            )
            
            individual_q_values = batch_current_q_values.to(self.device)
            individual_best_q_values = batch_current_best_q_values.to(self.device)
            
            # TD loss
            td_target = rewards_tensor + self.gamma * next_joint_q_value
            td_loss = self.criterion(joint_q_value, td_target.detach())
            
            # Optimality loss
            best_sum_q_i = individual_best_q_values.sum(dim=1, keepdim=True)
            opt_loss = self.criterion(best_joint_q_value.detach(), v_value + best_sum_q_i)
            
            # nopt loss
            sum_q_i = individual_q_values.sum(dim=1, keepdim=True)
            nopt_residual = sum_q_i - joint_q_value.detach() + v_value
            nopt_loss = torch.mean(torch.relu(-nopt_residual) ** 2)
            
            loss = td_loss + opt_loss + nopt_loss
            
            # 反向传播
            self.mixing_optimizer.zero_grad()
            for agent_name in self.agent_optimizer:
                self.agent_optimizer[agent_name].zero_grad()
            loss.backward()
            self.mixing_optimizer.step()
            for agent_name in self.agent_optimizer:
                self.agent_optimizer[agent_name].step()
            
            # 低频率更新 Shapley values
            if self.shapley_update_counter % self.shapley_update_interval == 0:
                self._update_shapley_values(batch_current_q_values)
            
            if self.shapley_update_counter % 10 == 0:
                shapley_str = ", ".join([f"A{i}:{self.cached_shapley_values[str(i)]:.3f}" 
                                        for i in range(self.agent_num)])
                logger.info(f"Hybrid loss: {loss.item():.4f}, Shapley: [{shapley_str}]")
    
    def _update_shapley_values(self, agent_q_values: torch.Tensor):
        """使用 Lovasz 扩展快速计算 Shapley values"""
        batch_size = agent_q_values.shape[0]
        agent_q_values = agent_q_values.detach()
        
        # 在 w = 0.5 处计算梯度
        w = torch.full(
            (batch_size, self.agent_num),
            0.5,
            device=self.device,
            requires_grad=True
        )
        
        q_tot = self.shapley_network(agent_q_values.to(self.device), w)
        q_tot.sum().backward()
        
        # 更新缓存
        mean_shapley = w.grad.mean(dim=0)
        shapley_sum = mean_shapley.sum().item()
        
        for i in range(self.agent_num):
            # 归一化
            self.cached_shapley_values[str(i)] = mean_shapley[i].item() / (shapley_sum + 1e-6)
    
    def update_target_networks(self):
        """更新目标网络"""
        with self.network_lock:
            for i in range(self.agent_num):
                agent_name = str(i)
                self.q_target_agent[agent_name].load_state_dict(
                    self.q_eval_agent[agent_name].state_dict()
                )
            self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        logger.info("Hybrid target networks updated")
