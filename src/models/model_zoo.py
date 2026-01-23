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
