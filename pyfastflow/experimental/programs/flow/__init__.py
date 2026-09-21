"""Compatibility imports for the promoted flow Programs."""

from pyfastflow.flow import (
    SFDFlowProgram, MFDFlowProgram,
    build_sfd_flow_program, build_mfd_flow_program,
)

__all__ = [
    "MFDFlowProgram", "SFDFlowProgram",
    "build_mfd_flow_program", "build_sfd_flow_program",
]
