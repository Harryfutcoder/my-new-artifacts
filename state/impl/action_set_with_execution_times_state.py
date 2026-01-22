import math
from collections import Counter
from collections import defaultdict
from typing import List, Dict, Union, Any, Tuple

from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from action.web_action import WebAction
from exceptions import WebtestException
from state.web_state import WebState


class ActionSetWithExecutionTimesState(WebState):
    def __init__(self, actions: List[WebAction], url: str) -> None:
        self.action_dict: Dict[WebAction, Dict[str, Union[int, WebState]]] = {
            key: {'execution_time': 0, 'child_state': None} for key in actions}
        self.action_execution_time_histogram: List[int] = [0] * 10
        self.action_execution_time_histogram[0] = len(self.action_dict)
        self.url: str = url
        self.sim_dic = defaultdict(float)

        self.global_url_list = set()
        self.global_form_list = set()
        self.global_tag_list = set()

    def get_action_list(self) -> List[WebAction]:
        action_list = list(self.action_dict.keys())
        action_list.sort()
        return action_list

    def get_action_detailed_data(self) -> Tuple[Dict[WebAction, Any], Any]:
        return self.action_dict, self.action_execution_time_histogram

    def update_action_execution_time(self, action: WebAction) -> None:
        if action in self.action_dict:
            self.action_dict[action]['execution_time'] += 1
            execution_time = self.action_dict[action]['execution_time']
            if execution_time < 10:
                self.action_execution_time_histogram[execution_time - 1] -= 1
                self.action_execution_time_histogram[execution_time] += 1
        else:
            raise WebtestException("The action is not exist in the state")

    def update_transition_information(self, action: WebAction, new_state: 'WebState') -> None:
        if action in self.action_dict:
            self.action_dict[action]['child_state'] = new_state
        else:
            raise WebtestException("The action is not exist in the state")

    def cosine_similarity(self, X: List[int], Y: List[int]) -> float:
        dot_product = sum(x * y for x, y in zip(X, Y))
        magnitude_X = math.sqrt(sum(x ** 2 for x in X))
        magnitude_Y = math.sqrt(sum(y ** 2 for y in Y))

        if magnitude_X == 0 or magnitude_Y == 0:
            return 0
        return float(dot_product) / (magnitude_X * magnitude_Y)

    def convert_action_to_vector(self, actions: List[WebAction]) -> Dict[str, list[int]]:
        action_counts = defaultdict(Counter)
        for action in actions:
            if isinstance(action, ClickAction):
                addition_info = action.addition_info
                action_counts[action.action_type][addition_info] += 1
                if action.action_type == 'redirect':
                    self.global_url_list.add(addition_info)
                elif action.action_type == 'submit':
                    self.global_form_list.add(addition_info)
                else:
                    self.global_tag_list.add(addition_info)
            elif isinstance(action, RandomInputAction):
                action_counts['default']['random_input'] += 1
            elif isinstance(action, RandomSelectAction):
                action_counts['default']['random_select'] += 1
        global_url_list_sorted = sorted(self.global_url_list)
        global_form_list_sorted = sorted(self.global_form_list)
        global_tag_list_sorted = sorted(self.global_tag_list)

        result = {}
        for action_type, addition_info_counter in action_counts.items():
            if action_type == "redirect":
                result[action_type] = [
                    addition_info_counter.get(href, 0) for href in global_url_list_sorted
                ]
            elif action_type == "submit":
                result[action_type] = [
                    addition_info_counter.get(count, 0) for count in global_form_list_sorted
                ]
            else:
                result[action_type] = [
                    addition_info_counter.get(tag_name, 0) for tag_name in global_tag_list_sorted
                ]
        return result

    def similarity(self, other: WebState) -> float:
        if not isinstance(other, ActionSetWithExecutionTimesState):
            return 0
        if not self.sim_dic.__contains__(other):
            vector_self = self.convert_action_to_vector(self.get_action_list())
            vector_other = self.convert_action_to_vector(other.get_action_list())
            redirect_similarity = 0 if not vector_self.get('redirect') or not vector_other.get('redirect') \
                else float(self.cosine_similarity(vector_self['redirect'], vector_other['redirect']))
            submit_similarity = 0 if not vector_self.get('submit') or not vector_other.get('submit') \
                else float(self.cosine_similarity(vector_self['submit'], vector_other['submit']))

            default_similarity = 0 if not vector_self.get('default') or not vector_other.get('default') \
                else float(self.cosine_similarity(vector_self['default'], vector_other['default']))

            weights = {
                'redirect': 0.75,
                'submit': 0.25,
                'default': 0.1
            }
            overall_similarity = float(weights['redirect'] * redirect_similarity +
                                  weights['submit'] * submit_similarity +
                                  weights['default'] * default_similarity)
            self.sim_dic[other] = overall_similarity
            other.sim_dic[self] = overall_similarity
        return self.sim_dic[other]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ActionSetWithExecutionTimesState):
            if not hasattr(self, 'action_dict') or not hasattr(other, 'action_dict'):
                return False
            if not hasattr(self, 'url') or not hasattr(other, 'url'):
                return False
            return (self.action_dict.keys() == other.action_dict.keys()) and (self.url == other.url)
        return False

    def __hash__(self) -> int:
        hash_value = 0
        if hasattr(self, 'action_dict'):
            for action in self.action_dict.keys():
                hash_value += hash(action)
        if hasattr(self, 'url'):
            hash_value += hash(self.url)
        return hash_value

    def __lt__(self, other: object) -> bool:
        if isinstance(other, ActionSetWithExecutionTimesState):
            return hash(self) < hash(other)
        else:
            return type(self).__name__ < type(other).__name__

    def __str__(self) -> str:
        return f"ActionSetWithExecutionTimesState(action_number={len(self.action_dict)}, action_execution_time_histogram={self.action_execution_time_histogram}, url={self.url})"
