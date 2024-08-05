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

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: jetstream/core/proto/jetstream.proto
# Protobuf Python Version: 5.26.1
# pylint: disable=all
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n$jetstream/core/proto/jetstream.proto\x12\x0fjetstream_proto"\xfc\x02\n\rDecodeRequest\x12\x12\n\nmax_tokens\x18\x04 \x01(\x05\x12\x42\n\x0ctext_content\x18\x05 \x01(\x0b\x32*.jetstream_proto.DecodeRequest.TextContentH\x00\x12\x44\n\rtoken_content\x18\x06 \x01(\x0b\x32+.jetstream_proto.DecodeRequest.TokenContentH\x00\x12;\n\x08metadata\x18\x07 \x01(\x0b\x32\'.jetstream_proto.DecodeRequest.MetadataH\x01\x1a\x1b\n\x0bTextContent\x12\x0c\n\x04text\x18\x01 \x01(\t\x1a!\n\x0cTokenContent\x12\x11\n\ttoken_ids\x18\x01 \x03(\x05\x1a\x1e\n\x08Metadata\x12\x12\n\nstart_time\x18\x01 \x01(\x02\x42\t\n\x07\x63ontentB\x13\n\x11metadata_optionalJ\x04\x08\x01\x10\x02J\x04\x08\x02\x10\x03J\x04\x08\x03\x10\x04"\xcb\x02\n\x0e\x44\x65\x63odeResponse\x12I\n\x0finitial_content\x18\x02 \x01(\x0b\x32..jetstream_proto.DecodeResponse.InitialContentH\x00\x12G\n\x0estream_content\x18\x03 \x01(\x0b\x32-.jetstream_proto.DecodeResponse.StreamContentH\x00\x1a\x10\n\x0eInitialContent\x1a\x81\x01\n\rStreamContent\x12\x45\n\x07samples\x18\x01 \x03(\x0b\x32\x34.jetstream_proto.DecodeResponse.StreamContent.Sample\x1a)\n\x06Sample\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x11\n\ttoken_ids\x18\x02 \x03(\x05\x42\t\n\x07\x63ontentJ\x04\x08\x01\x10\x02"\x14\n\x12HealthCheckRequest"&\n\x13HealthCheckResponse\x12\x0f\n\x07is_live\x18\x01 \x01(\x08\x32\xb9\x01\n\x0cOrchestrator\x12M\n\x06\x44\x65\x63ode\x12\x1e.jetstream_proto.DecodeRequest\x1a\x1f.jetstream_proto.DecodeResponse"\x00\x30\x01\x12Z\n\x0bHealthCheck\x12#.jetstream_proto.HealthCheckRequest\x1a$.jetstream_proto.HealthCheckResponse"\x00\x62\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, "jetstream.core.proto.jetstream_pb2", _globals
)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals["_DECODEREQUEST"]._serialized_start = 58
  _globals["_DECODEREQUEST"]._serialized_end = 438
  _globals["_DECODEREQUEST_TEXTCONTENT"]._serialized_start = 294
  _globals["_DECODEREQUEST_TEXTCONTENT"]._serialized_end = 321
  _globals["_DECODEREQUEST_TOKENCONTENT"]._serialized_start = 323
  _globals["_DECODEREQUEST_TOKENCONTENT"]._serialized_end = 356
  _globals["_DECODEREQUEST_METADATA"]._serialized_start = 358
  _globals["_DECODEREQUEST_METADATA"]._serialized_end = 388
  _globals["_DECODERESPONSE"]._serialized_start = 441
  _globals["_DECODERESPONSE"]._serialized_end = 772
  _globals["_DECODERESPONSE_INITIALCONTENT"]._serialized_start = 607
  _globals["_DECODERESPONSE_INITIALCONTENT"]._serialized_end = 623
  _globals["_DECODERESPONSE_STREAMCONTENT"]._serialized_start = 626
  _globals["_DECODERESPONSE_STREAMCONTENT"]._serialized_end = 755
  _globals["_DECODERESPONSE_STREAMCONTENT_SAMPLE"]._serialized_start = 714
  _globals["_DECODERESPONSE_STREAMCONTENT_SAMPLE"]._serialized_end = 755
  _globals["_HEALTHCHECKREQUEST"]._serialized_start = 774
  _globals["_HEALTHCHECKREQUEST"]._serialized_end = 794
  _globals["_HEALTHCHECKRESPONSE"]._serialized_start = 796
  _globals["_HEALTHCHECKRESPONSE"]._serialized_end = 834
  _globals["_ORCHESTRATOR"]._serialized_start = 837
  _globals["_ORCHESTRATOR"]._serialized_end = 1022
# @@protoc_insertion_point(module_scope)
