#!/bin/bash
export PYTHONPATH="/app"
exec python commonroad-docker-submission/planner/main_core1.py &
exec python commonroad-docker-submission/planner/main_core2.py
