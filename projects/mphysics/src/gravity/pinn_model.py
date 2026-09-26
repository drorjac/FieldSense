"""Compatibility shim: the PINN code lives in ``pinn.py``."""

from pinn import (PINN, data_loss, derivative, initial_condition_loss,  # noqa: F401
                  physics_loss)
