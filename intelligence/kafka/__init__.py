"""Kafka integration package for process event streaming."""

from intelligence.kafka.producer import ProcessEventKafkaProducer
from intelligence.kafka.consumer import ProcessEventKafkaConsumer
from intelligence.kafka.scope import ScopeManager

__all__ = ["ProcessEventKafkaProducer", "ProcessEventKafkaConsumer", "ScopeManager"]

