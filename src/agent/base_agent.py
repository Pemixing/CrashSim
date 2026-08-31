from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    msg_type: str
    payload: Dict[str, Any]
    trace_id: str


class BaseAgent(ABC):
    agent_name: str

    @abstractmethod
    def run(self, message: AgentMessage) -> AgentMessage:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError
