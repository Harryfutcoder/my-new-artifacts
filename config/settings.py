import os.path
import random
import sys

import yaml

from config.cli_options import cli_options


class Settings:
    def __init__(self) -> None:
        self.settings_path = cli_options.settings
        self.output_path = cli_options.output
        self.model_path = cli_options.model_path
        self.restart_interval = cli_options.restart_interval
        self.continuous_restart_threshold = cli_options.continuous_restart_threshold
        self.enable_screen_shot = cli_options.enable_screen_shot
        self.profile = cli_options.profile
        self.session = cli_options.session
        self.agent_num = cli_options.agent_num
        self.record_interval = None
        self.alive_time = None
        self.page_load_timeout = None
        self.browser_path = None
        self.browser_data_path = None
        self.driver_path = None
        self.resources_path = None
        self.entry_url = None
        self.domains = None
        self.browser_arguments = None
        self.action_detector = None
        self.state = None
        self.agent = None

    def load_settings(self) -> None:
        with open(self.settings_path, 'r') as f:
            settings_data = yaml.safe_load(f)
            if self.output_path is None:
                self.output_path = settings_data['default_output_path']
            if self.profile is None:
                self.profile = settings_data['default_profile']
            if self.session is None:
                self.session = settings_data['default_session']
            if self.model_path is None:
                self.model_path = settings_data['default_model_path']
            if self.restart_interval is None:
                self.restart_interval = settings_data['default_restart_interval']
            if self.continuous_restart_threshold is None:
                self.continuous_restart_threshold = settings_data['default_continuous_restart_threshold']
            if self.enable_screen_shot is None:
                self.enable_screen_shot = settings_data['default_enable_screen_shot']
            if self.profile not in settings_data['profiles']:
                print(f"Profile \"{self.profile}\" not exist", file=sys.stderr)
                sys.exit(1)
            if self.session == settings_data['default_session']:
                folder_name = self.profile + "-" + self.session + "-" + format(random.randint(0, 0xFFFFFF),
                                                                               '06x')
            else:
                folder_name = self.profile + "-" + self.session
            output_path = os.path.join(self.output_path, folder_name)
            while True:
                if os.path.exists(output_path):
                    if self.session == settings_data['default_session']:
                        folder_name = self.profile + "-" + self.session + "-" + format(random.randint(0, 0xFFFFFF),
                                                                                       '06x')
                    else:
                        folder_name = self.profile + "-" + self.session
                    output_path = os.path.join(self.output_path, folder_name)
                else:
                    break
            self.output_path = output_path
            self.agent_num = settings_data['profiles'][self.profile]['agent_num']
            self.record_interval = settings_data['profiles'][self.profile]['record_interval']
            self.alive_time = settings_data['profiles'][self.profile]['alive_time']
            self.page_load_timeout = settings_data['profiles'][self.profile]['page_load_timeout']
            self.browser_path = settings_data['profiles'][self.profile]['browser_path']
            self.browser_data_path = settings_data['profiles'][self.profile]['browser_data_path']
            self.driver_path = settings_data['profiles'][self.profile]['driver_path']
            self.resources_path = settings_data['profiles'][self.profile]['resources_path']
            self.entry_url = settings_data['profiles'][self.profile]['entry_url']
            self.domains = settings_data['profiles'][self.profile]['domains']
            self.browser_arguments = settings_data['profiles'][self.profile]['browser_arguments']
            self.action_detector = settings_data['profiles'][self.profile]['action_detector']
            self.state = settings_data['profiles'][self.profile]['state']
            self.agent = settings_data['profiles'][self.profile]['agent']
            if self.agent['module'] == "agent.impl.drl_agent" and self.agent['class'] == "DRLagent":
                self.load_drl_agent_cli_options()
            elif self.agent['module'] == "agent.impl.q_learning_agent" and self.agent['class'] == "QLearningAgent":
                self.load_q_learning_agent_cli_options()
            else:
                self.load_multi_agent_cli_options()

    def load_multi_agent_cli_options(self) -> None:
        self.agent["params"]["alive_time"] = self.alive_time
        self.agent["params"]["agent_num"] = self.agent_num
        self.agent["params"]["entry_url"] = self.entry_url
        if cli_options.model_module is not None:
            self.agent["params"]["model_module"] = cli_options.model_module
        if cli_options.model_class is not None:
            self.agent["params"]["model_class"] = cli_options.model_class
        if cli_options.model_load_type is not None:
            self.agent["params"]["model_load_type"] = cli_options.model_load_type
        if cli_options.model_load_name is not None:
            self.agent["params"]["model_load_name"] = cli_options.model_load_name
        if cli_options.transformer_module is not None:
            self.agent["params"]["transformer_module"] = cli_options.transformer_module
        if cli_options.transformer_class is not None:
            self.agent["params"]["transformer_class"] = cli_options.transformer_class
        if cli_options.reward_function is not None:
            self.agent["params"]["reward_function"] = cli_options.reward_function
        if cli_options.stop_update is not None:
            self.agent["params"]["stop_update"] = cli_options.stop_update
        if cli_options.batch_size is not None:
            self.agent["params"]["batch_size"] = cli_options.batch_size
        if cli_options.learning_rate is not None:
            self.agent["params"]["learning_rate"] = cli_options.learning_rate
        if cli_options.gamma is not None:
            self.agent["params"]["gamma"] = cli_options.gamma
        if cli_options.max_random is not None:
            self.agent["params"]["max_random"] = cli_options.max_random
        if cli_options.min_random is not None:
            self.agent["params"]["min_random"] = cli_options.min_random
        if cli_options.min_random is not None:
            self.agent["params"]["min_random"] = cli_options.min_random
        if cli_options.update_target_interval is not None:
            self.agent["params"]["update_target_interval"] = cli_options.update_target_interval
        if cli_options.update_target_interval is not None:
            self.agent["params"]["update_network_interval"] = cli_options.update_network_interval
        if cli_options.update_mixing_network_interval is not None:
            self.agent["params"]["update_mixing_network_interval"] = cli_options.update_mixing_network_interval


    def load_drl_agent_cli_options(self) -> None:
        self.agent["params"]["alive_time"] = self.alive_time
        if cli_options.model_module is not None:
            self.agent["params"]["model_module"] = cli_options.model_module
        if cli_options.model_class is not None:
            self.agent["params"]["model_class"] = cli_options.model_class
        if cli_options.model_load_type is not None:
            self.agent["params"]["model_load_type"] = cli_options.model_load_type
        if cli_options.model_load_name is not None:
            self.agent["params"]["model_load_name"] = cli_options.model_load_name
        if cli_options.transformer_module is not None:
            self.agent["params"]["transformer_module"] = cli_options.transformer_module
        if cli_options.transformer_class is not None:
            self.agent["params"]["transformer_class"] = cli_options.transformer_class
        if cli_options.reward_function is not None:
            self.agent["params"]["reward_function"] = cli_options.reward_function
        if cli_options.stop_update is not None:
            self.agent["params"]["stop_update"] = cli_options.stop_update
        if cli_options.batch_size is not None:
            self.agent["params"]["batch_size"] = cli_options.batch_size
        if cli_options.learning_rate is not None:
            self.agent["params"]["learning_rate"] = cli_options.learning_rate
        if cli_options.gamma is not None:
            self.agent["params"]["gamma"] = cli_options.gamma
        if cli_options.max_random is not None:
            self.agent["params"]["max_random"] = cli_options.max_random
        if cli_options.min_random is not None:
            self.agent["params"]["min_random"] = cli_options.min_random
        if cli_options.min_random is not None:
            self.agent["params"]["min_random"] = cli_options.min_random
        if cli_options.update_target_interval is not None:
            self.agent["params"]["update_target_interval"] = cli_options.update_target_interval
        if cli_options.update_target_interval is not None:
            self.agent["params"]["update_network_interval"] = cli_options.update_network_interval


    def load_q_learning_agent_cli_options(self) -> None:
        if cli_options.agent_type is not None:
            self.agent["params"]["agent_type"] = cli_options.agent_type
        if cli_options.alpha is not None:
            self.agent["params"]["alpha"] = cli_options.alpha
        if cli_options.gamma is not None:
            self.agent["params"]["gamma"] = cli_options.gamma
        if cli_options.epsilon is not None:
            self.agent["params"]["epsilon"] = cli_options.epsilon
        if cli_options.initial_q_value is not None:
            self.agent["params"]["initial_q_value"] = cli_options.initial_q_value
        if cli_options.r_reward is not None:
            self.agent["params"]["r_reward"] = cli_options.r_reward
        if cli_options.r_penalty is not None:
            self.agent["params"]["r_penalty"] = cli_options.r_penalty
        if cli_options.max_sim_line is not None:
            self.agent["params"]["max_sim_line"] = cli_options.max_sim_line


settings = Settings()
settings.load_settings()
