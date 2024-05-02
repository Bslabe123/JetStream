import datetime
import functools
import humanize
from typing import Optional
import logging

import jax
import flax
import numpy as np


def get_hbm_memory_stats(text: str, live_arrays: bool=False, raw: bool=False, memory_profile=False) -> None:
  logging.info(text)
  fmt_size = functools.partial(humanize.naturalsize, binary=True)
  all_stats = []
  for d in jax.local_devices():
    stats = d.memory_stats()
    all_stats.append(stats)
    used = stats['bytes_in_use']
    limit = stats['bytes_limit']
    print(f"    Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")
  if live_arrays:
    array_shapes = [array.shape for array in jax.live_arrays()]
    logging.info(f"  live arrays number: {len(array_shapes)}")
    logging.info(f"  live arrays shapes: {array_shapes}")
  if raw:
    logging.info(all_stats)
  if memory_profile:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    jax.profiler.save_device_memory_profile(f"{timestamp}_{text}.prof")


def calculate_num_params_from_pytree(params):
  params_sizes = jax.tree_util.tree_map(jax.numpy.size, params)
  total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
  assert total_parameters >= 0
  return total_parameters


def calculate_total_bytes_across_chip(params):
  def calculate_bytes_across_chip(arr):
    return [shard.data.nbytes for shard in arr.addressable_shards]
  bytes_across_chips = jax.tree_util.tree_map(calculate_bytes_across_chip, params)
  num_chips = len(bytes_across_chips)
  total_bytes_across_chip = jax.tree_util.tree_reduce(lambda x, y: x + y, bytes_across_chips)
  bytes_per_chip = total_bytes_across_chip / num_chips
  return total_bytes_across_chip, bytes_per_chip, num_chips


def calculate_total_params_across_chip(params):
  def calculate_sizes_per_chip(arr):
    return [np.prod(shard.data.shape) for shard in arr.addressable_shards]
  sizes_across_chips = jax.tree_util.tree_map(calculate_sizes_per_chip, params)
  num_chips = len(sizes_across_chips)
  total_sizes_across_chips = jax.tree_util.tree_reduce(lambda x, y: x + y, sizes_across_chips)
  sizes_per_chip = total_sizes_across_chips / num_chips
  return total_sizes_across_chips, sizes_per_chip, num_chips


def calculate_bytes_from_pytree(params):
  params_bytes = jax.tree_util.tree_map(lambda x: x.nbytes, params)
  total_bytes = jax.tree_util.tree_reduce(lambda x, y: x + y, params_bytes)
  return total_bytes


def summarize_size_from_pytree(params):
  num_params = calculate_num_params_from_pytree(params)
  num_bytes = calculate_bytes_from_pytree(params)
  return num_params, num_bytes, num_bytes / num_params


def debug_kv_cache(kv_cache):
  """Debug KV Cache sizing and sharding across chips."""
  singler_layer_kv_cache = kv_cache["cache"]["decoder"]["layers_0"]["self_attention"]["AttentionOp_0"]
  for cache_key in singler_layer_kv_cache.keys():
    cache_element = singler_layer_kv_cache[cache_key]
    print(f"{cache_key=}")
    if isinstance(cache_element, flax.linen.spmd.LogicallyPartitioned):
      cache_element = cache_element.value
    jax.debug.print("  shape: {}", cache_element.shape)
    jax.debug.print("  sharding: {}", cache_element.sharding)
    total_logical_sizes, total_logical_bytes, _ = summarize_size_from_pytree(cache_element)
    total_sizes_across_chips, sizes_per_chip, num_chips = calculate_total_params_across_chip(cache_element)
    total_bytes_across_chip, bytes_per_chip, _ = calculate_total_bytes_across_chip(cache_element)
    jax.debug.print("  total_logical_sizes: {}", total_logical_sizes)
    jax.debug.print("  total_logical_bytes: {}", total_logical_bytes)
    jax.debug.print("  num_chips: {}", num_chips)
    jax.debug.print("  total_sizes_across_chips: {}", total_sizes_across_chips)
    jax.debug.print("  sizes_per_chip: {}", sizes_per_chip)
    jax.debug.print("  total_bytes_across_chip: {}", total_bytes_across_chip)
    jax.debug.print("  bytes_per_chip: {}", bytes_per_chip)
