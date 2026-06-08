// Copyright 2026 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <functional>

#include "gtest/gtest.h"
#include "ray/asio/instrumented_io_context.h"
#include "ray/gcs/grpc_services.h"

namespace ray {
namespace rpc {

namespace {

grpc::health::v1::HealthCheckResponse::ServingStatus Check(
    HealthCheckGrpcService &service) {
  grpc::health::v1::HealthCheckRequest request;
  grpc::health::v1::HealthCheckResponse reply;
  service.HandleCheck(
      request, &reply, [](Status, std::function<void()>, std::function<void()>) {});
  return reply.status();
}

}  // namespace

TEST(HealthCheckGrpcServiceTest, ReflectsHealthCheckFn) {
  instrumented_io_context io_service;
  bool healthy = true;
  HealthCheckGrpcService service(io_service, [&healthy]() { return healthy; });

  EXPECT_EQ(Check(service), grpc::health::v1::HealthCheckResponse::SERVING);

  healthy = false;
  EXPECT_EQ(Check(service), grpc::health::v1::HealthCheckResponse::NOT_SERVING);

  healthy = true;
  EXPECT_EQ(Check(service), grpc::health::v1::HealthCheckResponse::SERVING);
}

}  // namespace rpc
}  // namespace ray

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
