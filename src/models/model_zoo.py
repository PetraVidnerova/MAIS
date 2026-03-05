"""Registry of all available simulation model classes.

``model_zoo`` is a dictionary mapping model-name strings to their
corresponding class objects.  It is used by loader utilities (e.g.
``load_model_from_config``) to look up the correct class by name at runtime.

Available models:

* ``"ExtendedNetworkModel"`` –
  :class:`~models.extended_network_model.ExtendedNetworkModel` (Gillespie
  engine).
* ``"ExtendedDailyNetworkModel"`` –
  :class:`~models.extended_network_model.ExtendedDailyNetworkModel` (daily
  Gillespie engine).
* ``"ExtendedSequentialNetworkModel"`` –
  :class:`~models.extended_network_model.ExtendedSequentialNetworkModel`
  (sequential discrete-step engine).
* ``"TGMNetworkModel"`` –
  :class:`~models.extended_network_model.TGMNetworkModel` (multi-layer-graph
  engine).
* ``"SimulationDrivenModel"`` –
  :class:`~models.agent_based_network_model.SimulationDrivenModel` (agent-
  based plan engine; currently the primary supported model).
* ``"InfoSIRModel"`` –
  :class:`~models.agent_info_models.InfoSIRModel`.
* ``"InfoTippingModel"`` –
  :class:`~models.agent_info_models.InfoTippingModel`.
* ``"RumourModel"`` –
  :class:`~models.agent_info_models.RumourModel`.
* ``"RumourModelInfo"`` –
  :class:`~models.agent_info_models.RumourModelInfo`.
"""

# dictionary of available models
# at the moment SimulationDrivenModel is the only one supported

#from models.seirs import SEIRSModel, SEIRSNetworkModel
from models.extended_network_model import ExtendedNetworkModel, ExtendedDailyNetworkModel, ExtendedSequentialNetworkModel
from models.extended_network_model import TGMNetworkModel
# from models.seirs_extended import ExtendedNetworkModel as OldExtendedNetworkModel
from models.agent_based_network_model import SimulationDrivenModel
from models.agent_info_models import InfoSIRModel, InfoTippingModel, RumourModel, RumourModelInfo

model_zoo = {
    "ExtendedNetworkModel": ExtendedNetworkModel,
    "ExtendedDailyNetworkModel": ExtendedDailyNetworkModel,
    "ExtendedSequentialNetworkModel": ExtendedSequentialNetworkModel,
    # "OldExtendedNetworkModel": OldExtendedNetworkModel,
    #   "SEIRSNetworkModel": SEIRSNetworkModel,
    "TGMNetworkModel": TGMNetworkModel,
    "SimulationDrivenModel": SimulationDrivenModel,
    "InfoSIRModel": InfoSIRModel,
    "InfoTippingModel": InfoTippingModel,
    "RumourModel": RumourModel,
    "RumourModelInfo": RumourModelInfo
}
