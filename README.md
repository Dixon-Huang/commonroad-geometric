# Introduction

**commonroad-geometric** is a Python framework that facilitates deep-learning based research projects in the autonomous driving domain, e.g. related to behavior planning and state representation learning.

At its core, it provides a standardized interface for heterogeneous graph representations of traffic scenes using the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) framework.

The package aims to serve as a flexible framework that, without putting restrictions on potential research directions, minimizes the time spent on implementing boilerplate code. Through its object-oriented design with highly flexible and extendable class interfaces, it is meant to be imported via **pip install** and utilized in a plug-and-play manner.

---



<img src="img/sumo_sim_temporal_1.gif" width="420" height="330" />
<img src="img/DEU_Munich-1_104-step-0-to-step-400.gif" width="420" height="330" />
<img src="img/CR-v0-DEU_Munich-1_19-2022-09-21-172426-step-0-to-step-20000.gif" width="420" height="330" />
<img src="img/occupancy_predictive_training.gif" width="420" height="330" />

---

## Highlighted features

- A framework for [PyTorch Geometric-based](https://pytorch-geometric.readthedocs.io/) heterogeneous graph data extraction from traffic scenes and road networks supporting user-made feature computers and edge drawers.
- Built-in functionality for collection and storing of graph-based traffic datasets as [PyTorch datasets](https://pytorch.org/vision/stable/datasets.html).
- Build-in support for both supervised learning tasks as well as reinforcement learning.
- A [OpenGL-based](https://pyglet.readthedocs.io/en/latest/programming_guide/gl.html#) traffic renderer with customizable plugins facilitating real-time visualization and debugging (see video above).
- An [OpenStreetMap](https://wiki.openstreetmap.org/wiki/API) scraper offering virtually unlimited access to real-world road networks.
- An interface to a [SUMO-based](https://sumo.dlr.de/) traffic simulator enabling both automated scenario generation, dataset collection as well as closed-loop training of autonomous agents.

---

## High-level package architecture


<img src="img/crgeo_high_level_architecture.svg" width="900" style="margin: 0 auto; overflow: hidden; margin-bottom: 20px" />

---


# Getting started

The easiest way of getting familiar with the framework is to consult the [tutorial directory](https://gitlab.lrz.de/cps/commonroad-geometric/-/tree/master/tutorials), which contains a multitude of simple application demos that showcase the intended usage of the package. Also, additional learning resources can be found in the repository's [Wiki](https://gitlab.lrz.de/cps/commonroad-geometric/-/wikis/home) section.

### Research guidelines:

- It is highly recommended to incorporate the package's extendable [rendering capabilities](https://gitlab.lrz.de/cps/commonroad-geometric/-/blob/develop/tutorials/render_with_custom_plugin.py) as an integral part of your development workflow. This allows you to visualize what is going on in your experiment, greatly simplifying debugging efforts.
- If you ever find yourself in a situation where you have to modify the internals of this package while working on your research project, it probably means that commonroad-geometric is not flexible enough - please [create a corresponding GitLab issue](https://gitlab.lrz.de/cps/commonroad-geometric/-/issues/new?issue%5Bmilestone_id%5D=).

### Design principles and developer guidelines:

- Frequent use of abstraction classes for a modular and clean framework.
- Class interfaces should be understandable and well-documented. We use the Google style docstring format across the package (see [PyCharm](https://www.jetbrains.com/help/pycharm/creating-documentation-comments.html), [VSCode](https://github.com/NilsJPWerner/autoDocstring)).
- As a general rule, everything should be configurable and externalized through class APIs. While ensuring flexibility, however, make sure to provide reasonable defaults for most things to avoid unnecessary overhead for users.
- Add tutorial scripts to the `tutorials/` directory for easy testing, reviewing and showcasing of new functionality.
- Use [type hinting](https://docs.python.org/3/library/typing.html) everywhere - it enhances readability and makes the IDE developer experience a lot smoother. Perform [static type checking](https://gitlab.lrz.de/cps/commonroad-geometric/-/wikis/Static-Type-Checking-with-Mypy) with [mypy](https://github.com/python/mypy) (`pip install mypy` + `/commonroad-geometric$ mypy`) for easily discovering inconsistent typing (see [PyCharm extension](https://plugins.jetbrains.com/plugin/11086-mypy), [VS Code extension](https://marketplace.visualstudio.com/items?itemName=matangover.mypy)).
- Create issues for new tasks with appropriate priority and category labels as well as a corresponding branch.  Create a merge request to the develop branch afterwards.
- Adhere to [PEP8](https://www.python.org/dev/peps/pep-0008/) (except linewidths).
- Use private `_attributes` and `_methods` for hiding internal implementation details, as well as private `_Classes` for helper classes not supposed to be exposed to end users.

---

# Installation

The installation script [`scripts/create-dev-environment.sh`](scripts/create-dev-environment.sh) installs the commonroad-geometric package and all its dependencies into a conda environment:

Execute the script inside the directory which you want to use for your development environment.

Note: make sure that the CUDA versions are compatible with your setup.

Related wiki pages:
  - [Manual installation](https://gitlab.lrz.de/cps/commonroad-geometric/-/wikis/Manual-installation)
  - [Troubleshooting](https://gitlab.lrz.de/cps/commonroad-geometric/-/wikis/Troubleshooting)


### Note: Headless rendering
If you want to export the rendering frames without the animation window popping up, please use the command given below.
``` shell
echo "export PYGLET_HEADLESS=..." >> ~/.bashrc
```
You can replace `.bashrc` with `.zshrc`, if you use `zsh`

---

## Weights & Biases Integration

**commonroad-geometric** offers a built-in **Weights & Biases** integration for metric tracking of deep learning experiments.

### Setup guide
1. Create your account at [https://wandb.ai/](https://wandb.ai/).
2. Create your project at Wandb and you should see a quickstart guide for pytorch.
3. Copy the **api key** give in the quickstart guide and set it as the environment variable *WANDB_API_KEY*.
4. Copy the **project name** and set it as the environment variable  *PROJECT_NAME*.
5. Finally, set the *ENTITY_NAME* environment variable with your username or in the case of service accounts, the name of the configured [service
account](https://docs.wandb.ai/guides/technical-faq#what-is-a-service-account-and-why-is-it-useful).

### Dashboard demo

<img src="img/wandb_demo.png" width="700" style="margin: 0 auto; overflow: hidden; margin-bottom: 20px" />

---

# Hyperparameter optimization
By default both [**Weights & Biases Sweeps**](https://wandb.ai/site/sweeps?gclid=Cj0KCQjwxtSSBhDYARIsAEn0thQrVQdHrcH2u7Sld8KkGbuCmuNRve2ltzrssvyKx28W21dhJ69DsLAaAnjwEALw_wcB) and [**Optuna**](https://optuna.org/) are supported.

### W&B sweeps
- You can run the script [`scripts/sweeps_optimization.sh`](scripts/sweeps_optimization.sh) with the parameters for the script you want to run:
``` shell
./scripts/sweeps_optimization.sh -e "<conda-env-name>" -s "<python-file>" -w "<path-to-sweep-file>" -a "(optional) <command-line-arguments-to-script>" -n "<number-of-runs>"
```

For example:

``` shell
./scripts/sweeps_optimization.sh -e commonroad-3.8 -s tutorials/train_geometric_model/run.py -w ../sweep_configs/dummy_model.yaml -a "train --model DummyModel --no-render --optimizer wandb --epochs 1"

```

Consult the [`scripts/sweep.template.yaml`](scripts/sweep.template.yaml) template for an introduction to configuring the your optimization run.

- The **entity** and **project** name must be set in the **sweep.yaml** file.
- The metric specified in the yaml file **must** be logged to wandb with the exact same string.
- For demonstration, refer to the **tutorials/train_geometric_model/run.py** script and pass the parameter *--optimizer wandb* in the script arguments (**-a**).

### Optuna

- For optuna demonstration, refer to the **tutorials/train_geometric_model/run.py** script.


