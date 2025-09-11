import logging
import os
import tempfile
import time

from selenium.webdriver.chrome.options import Options

from data_collector.data_collector_multi_agent import DataCollectorMultiAgent
from data_collector.data_collector_single_agent import DataCollector
from config.log_config import LogConfig
from config.settings import settings
from web_test.webtest_single_agent import Webtest
from web_test.webtest_multi_agent import WebtestMultiAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(LogConfig.get_file_handler())


def configure_chrome_options():
    """配置 Chrome 选项，禁止下载、弹出窗口等行为"""
    chrome_options = Options()
    chrome_options.binary_location = settings.browser_path

    # 用户数据
    # chrome_options.add_argument("--user-data-dir=webtest/userdata")

    # 添加浏览器启动参数
    for argument in settings.browser_arguments:
        chrome_options.add_argument(argument)

    # 禁止下载
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": "/dev/null",  # 设置下载目录为无效路径
        "download.prompt_for_download": False,     # 禁止下载弹窗
        "download.directory_upgrade": False,       # 禁止升级下载目录
    })

    # 禁止弹出窗口
    chrome_options.add_argument("--disable-popup-blocking")

    # 禁止自动保存密码
    chrome_options.add_experimental_option("prefs", {
        "credentials_enable_service": False,       # 禁用密码保存服务
        "profile.password_manager_enabled": False  # 禁用密码管理器
    })

    # 禁用扩展和插件
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")

    return chrome_options


def main():
    os.makedirs(settings.output_path, exist_ok=True)
    os.makedirs(settings.model_path, exist_ok=True)
    os.environ['PATH'] = settings.browser_path + os.pathsep + os.environ['PATH']

    # 配置 Chrome 选项
    chrome_options = configure_chrome_options()

    if settings.agent_num == 1:

        webtest = Webtest(chrome_options)
        data_collector = DataCollector(webtest)
    else:
        webtest = WebtestMultiAgent(chrome_options)
        data_collector = DataCollectorMultiAgent(webtest)

    data_collector.start()
    webtest.start()
    time.sleep(settings.alive_time)

    webtest.stop()
    data_collector.stop()
    data_collector.join()


if __name__ == '__main__':
    main()