"""Verification script for real Kafka -> Transformer end-to-end pipeline."""

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.inference.engine import ProcessInferenceEngine
from intelligence.inference.anomaly import ProcessAnomalyAnalyzer
from intelligence.inference.replay import ProcessEventReplayer
from intelligence.kafka.producer import ProcessEventKafkaProducer
from intelligence.kafka.consumer import ProcessEventKafkaConsumer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def test_end_to_end_kafka():
    print("=" * 70)
    print("STEP 1: Initializing Transformer Engine & Anomaly Analyzer...")
    engine = ProcessInferenceEngine()
    analyzer = ProcessAnomalyAnalyzer(engine)
    replayer = ProcessEventReplayer()
    print(f"Loaded checkpoint: {engine.checkpoint_meta['checkpoint_name']}")
    print(f"Validation threshold (p95): {analyzer.thresholds['active_threshold']:.2f}")

    print("\nSTEP 2: Connecting Producer and Consumer to Kafka (localhost:9092)...")
    # Use unique group_id for this test run so it reads from beginning of topic or newly published
    test_group = f"test-eval-{int(time.time())}"
    consumer = ProcessEventKafkaConsumer(
        engine=engine,
        analyzer=analyzer,
        bootstrap_servers="localhost:9092",
        topic="process-events",
        group_id=test_group,
    )
    # Configure consumer to read latest messages
    connected_c = consumer.connect()
    assert connected_c, "Failed to connect consumer to Kafka"

    producer = ProcessEventKafkaProducer(
        bootstrap_servers="localhost:9092",
        topic="process-events",
        replayer=replayer,
    )
    connected_p = producer.connect()
    assert connected_p, "Failed to connect producer to Kafka"
    print("Both Kafka producer and consumer successfully connected!")

    # Start consumer in background
    consumer.start()

    print("\nSTEP 3: Publishing 12 events to Kafka topic 'process-events'...")
    # Generate 12 events, which includes synthetic anomaly at step 6
    events = producer.build_demonstration_stream(max_events=12)
    published_count = 0
    for ev in events:
        ok = producer.publish_event(ev, key=str(ev.get("seq_id")))
        if ok:
            published_count += 1
        time.sleep(0.05)

    print(f"Published {published_count} messages to Kafka topic 'process-events'.")

    print("\nSTEP 4: Waiting for Kafka consumer to pull and score events...")
    # Wait up to 10 seconds for consumer to process all messages
    start_wait = time.time()
    while consumer.total_events < published_count and (time.time() - start_wait) < 10.0:
        time.sleep(0.2)

    consumer.stop()
    producer.close()
    consumer.close()

    status = consumer.get_status()
    print("\n" + "=" * 70)
    print("PIPELINE TEST RESULTS:")
    print(f"- Messages published: {published_count}")
    print(f"- Messages consumed:  {consumer.total_events}")
    print(f"- Events scored:      {consumer.total_events}")
    print(f"- NORMAL count:       {consumer.normal_events}")
    print(f"- UNUSUAL count:      {consumer.unusual_events}")
    print(f"- Latest detection:   {consumer.latest_detection}")

    print("\nDetailed Event Scores from Kafka stream:")
    for ev in list(status["recent_events"])[::-1]:
        print(f"  Step {ev['seq_id']:2d} | {ev['user']:12s} | {ev['computer']:6s} | {ev['process']:6s} | {ev['event_type']:5s} | Status: {ev['status']:7s} | Score: {ev['score']}")

    assert published_count == 12, f"Expected 12 published, got {published_count}"
    assert consumer.total_events == 12, f"Expected 12 consumed, got {consumer.total_events}"
    assert consumer.unusual_events >= 1, "Expected at least 1 UNUSUAL event"
    print("\nEND-TO-END TEST PASSED! Real Kafka transport and Transformer scoring confirmed.")
    return status


if __name__ == "__main__":
    test_end_to_end_kafka()

