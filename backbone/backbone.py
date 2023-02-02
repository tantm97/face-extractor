# -*- coding: utf-8 -*-
# Standard library imports
import sys

# Local application imports
from face_extractor.backbone import iresnet

sys.path.append("../..")


def get_backbone(m_params):
    """Get a backbone for the face extractor.

    Args:
        m_params: A dictionary of model params.

    Returns:
        A backbone for the face extractor.
    """
    if m_params["backbone"] == "iresnet18":
        backbone = iresnet.iresnet18(fp16=m_params["fp16"])
    elif m_params["backbone"] == "iresnet34":
        backbone = iresnet.iresnet34(fp16=m_params["fp16"])
    elif m_params["backbone"] == "iresnet50":
        backbone = iresnet.iresnet50(fp16=m_params["fp16"])
    else:
        pass
    return backbone
