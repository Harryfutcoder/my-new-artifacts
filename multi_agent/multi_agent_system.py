import logging
import threading
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict

from action.web_action import WebAction
from config.log_config import LogConfig
from exceptions import NoActionsException
from state.web_state import WebState

logger = logging.getLogger(__name__)
logger.addHandler(LogConfig.get_file_handler())


class MultiAgentSystem(ABC):
    def __init__(self, params) -> None:
        super(MultiAgentSystem, self).__init__()
        self.prev_state_dict: Dict[str, Optional[WebState]] = {}
        self.prev_action_dict: Dict[str, Optional[WebAction]] = {}
        self.current_state_dict: Dict[str, Optional[WebState]] = {}
        self.current_html_dict: Dict[str, str] = {}
        self.prev_html_dict: Dict[str, str] = {}
        self.action_dict: Dict[WebAction, int] = {}
        self.state_dict: Dict[WebState, int] = {}
        self.url_count_dict: Dict[str, int] = {}
        self.transition_record_list: List[Tuple[Optional[WebState], WebAction, WebState]] = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.agent_num = params["agent_num"]
        self.url_count_dict[params["entry_url"]] = 9999  # 避免没有url重启

    @abstractmethod
    def get_action_algorithm(self, web_state: WebState, html: str, agent_name: str) -> WebAction:
        pass

    def get_restart_url(self, agent_name: str) -> str:
        return min(self.url_count_dict, key=self.url_count_dict.get)

    def get_action(self, web_state: WebState, html: str, agent_name: str, url: str, check_result: bool) -> WebAction:
        if agent_name not in self.current_state_dict:
            self.current_state_dict[agent_name] = None
            self.prev_state_dict[agent_name] = None
            self.prev_action_dict[agent_name] = None
            self.current_html_dict[agent_name] = ''
            self.prev_html_dict[agent_name] = ''

        actions = web_state.get_action_list()
        if len(actions) == 0:
            self.prev_action_dict[agent_name] = None
            raise NoActionsException("The state does not have any actions")
        self.transit(web_state, agent_name, url, check_result, html)

        choose_action = self.get_action_algorithm(web_state, html, agent_name)
        self.prev_action_dict[agent_name] = choose_action
        with self.lock:
            if choose_action in self.action_dict:
                self.action_dict[choose_action] += 1
            else:
                self.action_dict[choose_action] = 1
        return choose_action

    def transit(self, new_state: WebState, agent_name: str, url: str, check_result: bool, html) -> None:
        actions = new_state.get_action_list()

        with self .lock:
            for action in actions:
                if action not in self.action_dict:
                    self.action_dict[action] = 0

            if new_state in self.state_dict:
                self.state_dict[new_state] += 1
            else:
                self.state_dict[new_state] = 1

            self.prev_state_dict[agent_name] = self.current_state_dict[agent_name]
            self.current_state_dict[agent_name] = new_state
            self.prev_html_dict[agent_name] = self.current_html_dict[agent_name]
            self.current_html_dict[agent_name] = html

            if check_result:
                if url in self.url_count_dict:
                    self.url_count_dict[url] += 1
                else:
                    self.url_count_dict[url] = 1

            if self.prev_action_dict[agent_name] is not None and self.prev_state_dict[agent_name] is not None and \
                    self.current_state_dict[agent_name] is not None:
                self.transition_record_list.append((self.prev_state_dict[agent_name], self.prev_action_dict[agent_name],
                                                    self.current_state_dict[agent_name]))

    def restart_fail(self, agent_name: str, restart_url: str) -> None:
        with self.lock:
            self.url_count_dict[restart_url] += 1

    def deal_exception(self, agent_name):
        self.current_state_dict[agent_name] = None
        self.prev_state_dict[agent_name] = None
        self.prev_action_dict[agent_name] = None
        self.current_html_dict[agent_name] = ''
        self.prev_html_dict[agent_name] = ''

    def get_state(self, web_state: WebState) -> WebState:
        with self.lock:
            if web_state not in self.state_dict:
                self.state_dict[web_state] = 0
                return web_state

        for state in self.state_dict.keys():
            if state == web_state:
                return state
