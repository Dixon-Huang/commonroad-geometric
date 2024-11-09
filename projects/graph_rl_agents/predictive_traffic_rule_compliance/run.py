import multiprocessing
import sys
import os
import hydra
from omegaconf import OmegaConf
from commonroad_geometric.learning.reinforcement.project.hydra_rl_config import RLProjectConfig
from projects.graph_rl_agents.predictive_traffic_rule_compliance.project import PredictiveTrafficRuleComplianceProject
from projects.graph_rl_agents.predictive_traffic_rule_compliance.config.custom_hydra_rl_config import \
    CustomRLProjectConfig
import logging

logger = logging.getLogger(__name__)
sys.path.insert(0, os.getcwd())


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: RLProjectConfig) -> None:
    cfg_obj = OmegaConf.to_object(cfg)

    # Stage 1
    cfg_obj.warmstart = False
    project = PredictiveTrafficRuleComplianceProject(cfg=cfg_obj, stage=1)
    project.run(cfg_obj.cmd)
    logger.info("===============================================================")
    logger.info("====================== Stage 1 Completed ======================")
    logger.info("===============================================================")

    # # Stage 2
    # cfg_obj.warmstart = True
    # project = PredictiveTrafficRuleComplianceProject(cfg=cfg_obj, stage=2)
    # project.run(cfg_obj.cmd)
    # logger.info("Stage 2 completed successfully")
    # logger.info("===============================================================")
    # logger.info("====================== Stage 2 Completed ======================")
    # logger.info("===============================================================")
    #
    # # Stage 3
    # cfg_obj.warmstart = True
    # project = PredictiveTrafficRuleComplianceProject(cfg=cfg_obj, stage=3)
    # project.run(cfg_obj.cmd)
    # logger.info("Stage 3 completed successfully")
    # logger.info("===============================================================")
    # logger.info("====================== Stage 3 Completed ======================")
    # logger.info("===============================================================")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
