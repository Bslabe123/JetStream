# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from jetstream.core.proto import jetstream_pb2 as jetstream_dot_core_dot_proto_dot_jetstream__pb2

GRPC_GENERATED_VERSION = "1.65.4"
GRPC_VERSION = grpc.__version__
EXPECTED_ERROR_RELEASE = "1.66.0"
SCHEDULED_RELEASE_DATE = "August 6, 2024"
_version_not_supported = False

try:
  from grpc._utilities import first_version_is_lower

  _version_not_supported = first_version_is_lower(
      GRPC_VERSION, GRPC_GENERATED_VERSION
  )
except ImportError:
  _version_not_supported = True

if _version_not_supported:
  warnings.warn(
      f"The grpc package installed is at version {GRPC_VERSION},"
      + f" but the generated code in jetstream/core/proto/jetstream_pb2_grpc.py depends on"
      + f" grpcio>={GRPC_GENERATED_VERSION}."
      + f" Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}"
      + f" or downgrade your generated code using grpcio-tools<={GRPC_VERSION}."
      + f" This warning will become an error in {EXPECTED_ERROR_RELEASE},"
      + f" scheduled for release on {SCHEDULED_RELEASE_DATE}.",
      RuntimeWarning,
  )


class OrchestratorStub(object):
  """TODO: Merge this with main JetStream core once we settle on an API."""

  def __init__(self, channel):
    """Constructor.

    Args:
        channel: A grpc.Channel.
    """
    self.Decode = channel.unary_stream(
        "/jetstream_proto.Orchestrator/Decode",
        request_serializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.DecodeRequest.SerializeToString,
        response_deserializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.DecodeResponse.FromString,
        _registered_method=True,
    )
    self.HealthCheck = channel.unary_unary(
        "/jetstream_proto.Orchestrator/HealthCheck",
        request_serializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.HealthCheckRequest.SerializeToString,
        response_deserializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.HealthCheckResponse.FromString,
        _registered_method=True,
    )


class OrchestratorServicer(object):
  """TODO: Merge this with main JetStream core once we settle on an API."""

  def Decode(self, request, context):
    """Query LLM to generate text or tokens."""
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details("Method not implemented!")
    raise NotImplementedError("Method not implemented!")

  def HealthCheck(self, request, context):
    """Checks if the model server is live."""
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details("Method not implemented!")
    raise NotImplementedError("Method not implemented!")


def add_OrchestratorServicer_to_server(servicer, server):
  rpc_method_handlers = {
      "Decode": grpc.unary_stream_rpc_method_handler(
          servicer.Decode,
          request_deserializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.DecodeRequest.FromString,
          response_serializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.DecodeResponse.SerializeToString,
      ),
      "HealthCheck": grpc.unary_unary_rpc_method_handler(
          servicer.HealthCheck,
          request_deserializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.HealthCheckRequest.FromString,
          response_serializer=jetstream_dot_core_dot_proto_dot_jetstream__pb2.HealthCheckResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      "jetstream_proto.Orchestrator", rpc_method_handlers
  )
  server.add_generic_rpc_handlers((generic_handler,))
  server.add_registered_method_handlers(
      "jetstream_proto.Orchestrator", rpc_method_handlers
  )


# This class is part of an EXPERIMENTAL API.
class Orchestrator(object):
  """TODO: Merge this with main JetStream core once we settle on an API."""

  @staticmethod
  def Decode(
      request,
      target,
      options=(),
      channel_credentials=None,
      call_credentials=None,
      insecure=False,
      compression=None,
      wait_for_ready=None,
      timeout=None,
      metadata=None,
  ):
    return grpc.experimental.unary_stream(
        request,
        target,
        "/jetstream_proto.Orchestrator/Decode",
        jetstream_dot_core_dot_proto_dot_jetstream__pb2.DecodeRequest.SerializeToString,
        jetstream_dot_core_dot_proto_dot_jetstream__pb2.DecodeResponse.FromString,
        options,
        channel_credentials,
        insecure,
        call_credentials,
        compression,
        wait_for_ready,
        timeout,
        metadata,
        _registered_method=True,
    )

  @staticmethod
  def HealthCheck(
      request,
      target,
      options=(),
      channel_credentials=None,
      call_credentials=None,
      insecure=False,
      compression=None,
      wait_for_ready=None,
      timeout=None,
      metadata=None,
  ):
    return grpc.experimental.unary_unary(
        request,
        target,
        "/jetstream_proto.Orchestrator/HealthCheck",
        jetstream_dot_core_dot_proto_dot_jetstream__pb2.HealthCheckRequest.SerializeToString,
        jetstream_dot_core_dot_proto_dot_jetstream__pb2.HealthCheckResponse.FromString,
        options,
        channel_credentials,
        insecure,
        call_credentials,
        compression,
        wait_for_ready,
        timeout,
        metadata,
        _registered_method=True,
    )
