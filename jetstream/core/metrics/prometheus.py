# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains common functions for configuring Jetstream server metrics"""

import os
import shortuuid
import time
from prometheus_client import Gauge


class TPOTMetricsCollector:
  """Wrapper class for collecting TPOT per inference request and reporting TPOT averages"""

  window_size = 60
  max_elements = 100

  # timestamp, value
  tpot_list: list[tuple[float, float]] = [[]]

  def put(self, value: float):
    self.tpot_list.insert(0, (time.time(), value))
    del self.tpot_list[self.max_elements :]

  # Report the average TPOT over the last `max_elements` inference requests over the last `window_size` seconds
  def average(self) -> float:
    cur_time = time.time()
    samples = filter(
        lambda tpot: cur_time - self.window_size < tpot.index(0),
        self.tpot_list.copy(),
    )
    return sum(samples) / len(samples)


class JetstreamMetricsCollector:
  """Wrapper class should be used to assure all metrics have proper tags"""

  _id: str = os.getenv("HOSTNAME", shortuuid.uuid())
  tpot_metrics_collector: TPOTMetricsCollector = TPOTMetricsCollector()

  def __new__(cls):
    if not hasattr(cls, "instance"):
      cls.instance = super(JetstreamMetricsCollector, cls).__new__(cls)
    return cls.instance

  # Metric definitions
  _prefill_backlog = Gauge(
      name="jetstream_prefill_backlog_size",
      documentation="Size of prefill queue",
      labelnames=["id"],
  )
  _slots_available_percentage = Gauge(
      name="jetstream_slots_available_percentage",
      documentation="The percentage of available slots in decode batch",
      labelnames=["id", "idx"],
  )
  _average_tpot = Gauge(
      name="jetstream_slots_available_percentage",
      documentation="Average TPOT for requests over the past second",
      labelnames=["id"],
  )

  def get_prefill_backlog_metric(self):
    return self._prefill_backlog.labels(id=self._id)

  def get_slots_available_percentage_metric(self, idx: int):
    return self._slots_available_percentage.labels(id=self._id, idx=idx)

  def get_average_tpot_metric(self):
    return self._average_tpot.labels(id=self._id)
