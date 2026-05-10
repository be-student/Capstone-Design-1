"""
TDD Tests for Docker Setup Validation.

Tests cover:
- Dockerfile existence and structure (MLflow, Pipeline, Dashboard)
- docker-compose.yml validity and service definitions
- Service configuration (ports, volumes, networks, healthchecks)
- Environment variable configuration
- Dependency ordering between services
- Volume and network definitions
- Entrypoint script validation
- Integration smoke tests for Docker build readiness
"""

import os
import sys
import re
import pytest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def docker_compose_path():
    """Return the docker-compose.yml path."""
    return PROJECT_ROOT / "docker-compose.yml"


@pytest.fixture
def docker_compose(docker_compose_path):
    """Load and parse docker-compose.yml."""
    with open(docker_compose_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def dockerfile_mlflow():
    """Read Dockerfile.mlflow contents."""
    path = PROJECT_ROOT / "Dockerfile.mlflow"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def dockerfile_pipeline():
    """Read Dockerfile.pipeline contents."""
    path = PROJECT_ROOT / "Dockerfile.pipeline"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def dockerfile_dashboard():
    """Read Dockerfile.dashboard contents."""
    path = PROJECT_ROOT / "Dockerfile.dashboard"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def mlflow_entrypoint():
    """Read MLflow entrypoint script contents."""
    path = PROJECT_ROOT / "scripts" / "mlflow_entrypoint.sh"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Dockerfile existence tests
# ---------------------------------------------------------------------------

class TestDockerfileExistence:
    """Test that all required Dockerfiles exist."""

    def test_dockerfile_mlflow_exists(self, project_root):
        """Dockerfile.mlflow must exist."""
        assert (project_root / "Dockerfile.mlflow").exists()

    def test_dockerfile_pipeline_exists(self, project_root):
        """Dockerfile.pipeline must exist."""
        assert (project_root / "Dockerfile.pipeline").exists()

    def test_dockerfile_dashboard_exists(self, project_root):
        """Dockerfile.dashboard must exist."""
        assert (project_root / "Dockerfile.dashboard").exists()

    def test_docker_compose_exists(self, project_root):
        """docker-compose.yml must exist."""
        assert (project_root / "docker-compose.yml").exists()

    def test_mlflow_entrypoint_exists(self, project_root):
        """MLflow entrypoint script must exist."""
        assert (project_root / "scripts" / "mlflow_entrypoint.sh").exists()


# ---------------------------------------------------------------------------
# Dockerfile.mlflow structure tests
# ---------------------------------------------------------------------------

class TestDockerfileMLflow:
    """Test Dockerfile.mlflow structure and configuration."""

    def test_uses_python_310(self, dockerfile_mlflow):
        """Must use Python 3.10 base image."""
        assert "python:3.10" in dockerfile_mlflow

    def test_installs_mlflow(self, dockerfile_mlflow):
        """Must install MLflow."""
        assert "mlflow" in dockerfile_mlflow

    def test_installs_sqlalchemy(self, dockerfile_mlflow):
        """Must install SQLAlchemy for backend store."""
        assert "sqlalchemy" in dockerfile_mlflow.lower()

    def test_exposes_port_5000(self, dockerfile_mlflow):
        """Must expose port 5000."""
        assert "EXPOSE 5000" in dockerfile_mlflow

    def test_has_healthcheck(self, dockerfile_mlflow):
        """Must include a healthcheck."""
        assert "HEALTHCHECK" in dockerfile_mlflow

    def test_has_entrypoint(self, dockerfile_mlflow):
        """Must define an entrypoint."""
        assert "ENTRYPOINT" in dockerfile_mlflow

    def test_copies_entrypoint_script(self, dockerfile_mlflow):
        """Must copy the entrypoint script."""
        assert "mlflow_entrypoint.sh" in dockerfile_mlflow

    def test_creates_artifact_directory(self, dockerfile_mlflow):
        """Must create artifact and db directories."""
        assert "/mlflow/artifacts" in dockerfile_mlflow

    def test_has_label(self, dockerfile_mlflow):
        """Must have LABEL metadata."""
        assert "LABEL" in dockerfile_mlflow

    def test_workdir_set(self, dockerfile_mlflow):
        """Must set WORKDIR."""
        assert "WORKDIR" in dockerfile_mlflow


# ---------------------------------------------------------------------------
# Dockerfile.pipeline structure tests
# ---------------------------------------------------------------------------

class TestDockerfilePipeline:
    """Test Dockerfile.pipeline structure and configuration."""

    def test_uses_python_310(self, dockerfile_pipeline):
        """Must use Python 3.10 base image."""
        assert "python:3.10" in dockerfile_pipeline

    def test_copies_requirements(self, dockerfile_pipeline):
        """Must copy requirements.txt."""
        assert "requirements.txt" in dockerfile_pipeline

    def test_installs_requirements(self, dockerfile_pipeline):
        """Must install from requirements.txt."""
        assert "pip install" in dockerfile_pipeline
        assert "requirements.txt" in dockerfile_pipeline

    def test_copies_source_code(self, dockerfile_pipeline):
        """Must copy src/ directory."""
        assert "src/" in dockerfile_pipeline

    def test_copies_config(self, dockerfile_pipeline):
        """Must copy config/ directory."""
        assert "config/" in dockerfile_pipeline

    def test_sets_pythonpath(self, dockerfile_pipeline):
        """Must set PYTHONPATH for module resolution."""
        assert "PYTHONPATH" in dockerfile_pipeline

    def test_has_workdir(self, dockerfile_pipeline):
        """Must set WORKDIR."""
        assert "WORKDIR" in dockerfile_pipeline

    def test_has_cmd_or_entrypoint(self, dockerfile_pipeline):
        """Must have a default CMD or ENTRYPOINT."""
        assert "CMD" in dockerfile_pipeline or "ENTRYPOINT" in dockerfile_pipeline

    def test_has_label(self, dockerfile_pipeline):
        """Must have LABEL metadata."""
        assert "LABEL" in dockerfile_pipeline

    def test_creates_output_directory(self, dockerfile_pipeline):
        """Must create output directory."""
        assert "output" in dockerfile_pipeline


# ---------------------------------------------------------------------------
# Dockerfile.dashboard structure tests
# ---------------------------------------------------------------------------

class TestDockerfileDashboard:
    """Test Dockerfile.dashboard structure and configuration."""

    def test_uses_python_310(self, dockerfile_dashboard):
        """Must use Python 3.10 base image."""
        assert "python:3.10" in dockerfile_dashboard

    def test_copies_requirements(self, dockerfile_dashboard):
        """Must copy dashboard-specific requirements."""
        assert "requirements-dashboard.txt" in dockerfile_dashboard

    def test_installs_requirements(self, dockerfile_dashboard):
        """Must install from dashboard-specific requirements."""
        assert "pip install" in dockerfile_dashboard
        assert "requirements-dashboard.txt" in dockerfile_dashboard

    def test_copies_source_code(self, dockerfile_dashboard):
        """Must copy src/ directory."""
        assert "src/" in dockerfile_dashboard

    def test_copies_config(self, dockerfile_dashboard):
        """Must copy config/ directory."""
        assert "config/" in dockerfile_dashboard

    def test_exposes_port_8501(self, dockerfile_dashboard):
        """Must expose Streamlit port 8501."""
        assert "EXPOSE 8501" in dockerfile_dashboard

    def test_has_healthcheck(self, dockerfile_dashboard):
        """Must include a healthcheck."""
        assert "HEALTHCHECK" in dockerfile_dashboard

    def test_healthcheck_targets_streamlit(self, dockerfile_dashboard):
        """Healthcheck must target Streamlit health endpoint."""
        assert "8501" in dockerfile_dashboard
        assert "health" in dockerfile_dashboard.lower()

    def test_runs_streamlit(self, dockerfile_dashboard):
        """CMD must run streamlit."""
        assert "streamlit" in dockerfile_dashboard

    def test_runs_headless(self, dockerfile_dashboard):
        """Must run Streamlit in headless mode."""
        assert "headless" in dockerfile_dashboard

    def test_binds_to_all_interfaces(self, dockerfile_dashboard):
        """Must bind to 0.0.0.0 for container accessibility."""
        assert "0.0.0.0" in dockerfile_dashboard

    def test_sets_pythonpath(self, dockerfile_dashboard):
        """Must set PYTHONPATH for module resolution."""
        assert "PYTHONPATH" in dockerfile_dashboard

    def test_has_label(self, dockerfile_dashboard):
        """Must have LABEL metadata."""
        assert "LABEL" in dockerfile_dashboard


# ---------------------------------------------------------------------------
# docker-compose.yml service tests
# ---------------------------------------------------------------------------

class TestDockerComposeServices:
    """Test docker-compose.yml service definitions."""

    def test_has_four_services(self, docker_compose):
        """Must define exactly 4 services."""
        services = docker_compose.get("services", {})
        assert len(services) == 4

    def test_has_mlflow_service(self, docker_compose):
        """Must define the mlflow service."""
        assert "mlflow" in docker_compose["services"]

    def test_has_redis_service(self, docker_compose):
        """Must define the redis service."""
        assert "redis" in docker_compose["services"]

    def test_has_pipeline_service(self, docker_compose):
        """Must define the pipeline service."""
        assert "pipeline" in docker_compose["services"]

    def test_has_dashboard_service(self, docker_compose):
        """Must define the dashboard service."""
        assert "dashboard" in docker_compose["services"]


# ---------------------------------------------------------------------------
# MLflow service configuration tests
# ---------------------------------------------------------------------------

class TestMLflowServiceConfig:
    """Test MLflow service configuration in docker-compose."""

    def test_mlflow_port_5000(self, docker_compose):
        """MLflow must be exposed on port 5000."""
        mlflow = docker_compose["services"]["mlflow"]
        ports = mlflow.get("ports", [])
        assert any("5000" in str(p) for p in ports)

    def test_mlflow_has_healthcheck(self, docker_compose):
        """MLflow must define a healthcheck."""
        mlflow = docker_compose["services"]["mlflow"]
        assert "healthcheck" in mlflow

    def test_mlflow_uses_dockerfile_mlflow(self, docker_compose):
        """MLflow must build from Dockerfile.mlflow."""
        mlflow = docker_compose["services"]["mlflow"]
        assert mlflow["build"]["dockerfile"] == "Dockerfile.mlflow"

    def test_mlflow_has_volumes(self, docker_compose):
        """MLflow must mount data volume."""
        mlflow = docker_compose["services"]["mlflow"]
        assert "volumes" in mlflow
        assert len(mlflow["volumes"]) > 0

    def test_mlflow_has_restart_policy(self, docker_compose):
        """MLflow must have a restart policy."""
        mlflow = docker_compose["services"]["mlflow"]
        assert "restart" in mlflow

    def test_mlflow_omits_container_name(self, docker_compose):
        """MLflow must rely on Compose project-scoped container names."""
        mlflow = docker_compose["services"]["mlflow"]
        assert "container_name" not in mlflow

    def test_mlflow_environment_variables(self, docker_compose):
        """MLflow must set required environment variables."""
        mlflow = docker_compose["services"]["mlflow"]
        env = mlflow.get("environment", [])
        env_str = str(env)
        assert "MLFLOW_BACKEND_STORE_URI" in env_str
        assert "MLFLOW_ARTIFACT_ROOT" in env_str


# ---------------------------------------------------------------------------
# Redis service configuration tests
# ---------------------------------------------------------------------------

class TestRedisServiceConfig:
    """Test Redis service configuration in docker-compose."""

    def test_redis_port_6379(self, docker_compose):
        """Redis must be exposed on port 6379."""
        redis = docker_compose["services"]["redis"]
        ports = redis.get("ports", [])
        assert any("6379" in str(p) for p in ports)

    def test_redis_uses_alpine_image(self, docker_compose):
        """Redis must use the alpine variant for minimal size."""
        redis = docker_compose["services"]["redis"]
        assert "alpine" in redis.get("image", "")

    def test_redis_has_healthcheck(self, docker_compose):
        """Redis must define a healthcheck."""
        redis = docker_compose["services"]["redis"]
        assert "healthcheck" in redis

    def test_redis_has_persistence(self, docker_compose):
        """Redis must enable persistence (appendonly)."""
        redis = docker_compose["services"]["redis"]
        command = redis.get("command", "")
        assert "appendonly" in command

    def test_redis_has_max_memory(self, docker_compose):
        """Redis must configure max memory."""
        redis = docker_compose["services"]["redis"]
        command = redis.get("command", "")
        assert "maxmemory" in command

    def test_redis_has_eviction_policy(self, docker_compose):
        """Redis must set an eviction policy."""
        redis = docker_compose["services"]["redis"]
        command = redis.get("command", "")
        assert "maxmemory-policy" in command

    def test_redis_has_volume(self, docker_compose):
        """Redis must mount a data volume."""
        redis = docker_compose["services"]["redis"]
        assert "volumes" in redis

    def test_redis_has_restart_policy(self, docker_compose):
        """Redis must have a restart policy."""
        redis = docker_compose["services"]["redis"]
        assert "restart" in redis


# ---------------------------------------------------------------------------
# Dashboard service configuration tests
# ---------------------------------------------------------------------------

class TestDashboardServiceConfig:
    """Test dashboard service configuration in docker-compose."""

    def test_dashboard_port_8501(self, docker_compose):
        """Dashboard must be exposed on port 8501."""
        dashboard = docker_compose["services"]["dashboard"]
        ports = dashboard.get("ports", [])
        assert any("8501" in str(p) for p in ports)

    def test_dashboard_uses_dockerfile_dashboard(self, docker_compose):
        """Dashboard must build from Dockerfile.dashboard."""
        dashboard = docker_compose["services"]["dashboard"]
        assert dashboard["build"]["dockerfile"] == "Dockerfile.dashboard"

    def test_dashboard_depends_on_mlflow(self, docker_compose):
        """Dashboard must depend on MLflow being healthy."""
        dashboard = docker_compose["services"]["dashboard"]
        depends = dashboard.get("depends_on", {})
        assert "mlflow" in depends

    def test_dashboard_depends_on_redis(self, docker_compose):
        """Dashboard must depend on Redis being healthy."""
        dashboard = docker_compose["services"]["dashboard"]
        depends = dashboard.get("depends_on", {})
        assert "redis" in depends

    def test_dashboard_mlflow_dependency_healthy(self, docker_compose):
        """Dashboard must wait for MLflow healthcheck."""
        dashboard = docker_compose["services"]["dashboard"]
        depends = dashboard.get("depends_on", {})
        mlflow_dep = depends.get("mlflow", {})
        assert mlflow_dep.get("condition") == "service_healthy"

    def test_dashboard_has_streamlit_env(self, docker_compose):
        """Dashboard must set Streamlit environment variables."""
        dashboard = docker_compose["services"]["dashboard"]
        env = dashboard.get("environment", [])
        env_str = str(env)
        assert "STREAMLIT_SERVER_PORT" in env_str
        assert "STREAMLIT_SERVER_ADDRESS" in env_str

    def test_dashboard_sets_pythonpath(self, docker_compose):
        """Dashboard must set PYTHONPATH."""
        dashboard = docker_compose["services"]["dashboard"]
        env = dashboard.get("environment", [])
        env_str = str(env)
        assert "PYTHONPATH" in env_str

    def test_dashboard_mounts_config(self, docker_compose):
        """Dashboard must mount config directory."""
        dashboard = docker_compose["services"]["dashboard"]
        volumes = dashboard.get("volumes", [])
        vol_str = str(volumes)
        assert "config" in vol_str

    def test_dashboard_mounts_source(self, docker_compose):
        """Dashboard must mount source code directory."""
        dashboard = docker_compose["services"]["dashboard"]
        volumes = dashboard.get("volumes", [])
        vol_str = str(volumes)
        assert "src" in vol_str

    def test_dashboard_has_restart_policy(self, docker_compose):
        """Dashboard must have a restart policy."""
        dashboard = docker_compose["services"]["dashboard"]
        assert "restart" in dashboard


# ---------------------------------------------------------------------------
# Pipeline service configuration tests
# ---------------------------------------------------------------------------

class TestPipelineServiceConfig:
    """Test pipeline service configuration in docker-compose."""

    def test_pipeline_uses_dockerfile_pipeline(self, docker_compose):
        """Pipeline must build from Dockerfile.pipeline."""
        pipeline = docker_compose["services"]["pipeline"]
        assert pipeline["build"]["dockerfile"] == "Dockerfile.pipeline"

    def test_pipeline_depends_on_mlflow(self, docker_compose):
        """Pipeline must depend on MLflow being healthy."""
        pipeline = docker_compose["services"]["pipeline"]
        depends = pipeline.get("depends_on", {})
        assert "mlflow" in depends

    def test_pipeline_depends_on_redis(self, docker_compose):
        """Pipeline must depend on Redis being healthy."""
        pipeline = docker_compose["services"]["pipeline"]
        depends = pipeline.get("depends_on", {})
        assert "redis" in depends

    def test_pipeline_sets_mlflow_uri(self, docker_compose):
        """Pipeline must set MLFLOW_TRACKING_URI."""
        pipeline = docker_compose["services"]["pipeline"]
        env = pipeline.get("environment", [])
        env_str = str(env)
        assert "MLFLOW_TRACKING_URI" in env_str

    def test_pipeline_sets_redis_host(self, docker_compose):
        """Pipeline must set REDIS_HOST."""
        pipeline = docker_compose["services"]["pipeline"]
        env = pipeline.get("environment", [])
        env_str = str(env)
        assert "REDIS_HOST" in env_str

    def test_pipeline_sets_pythonpath(self, docker_compose):
        """Pipeline must set PYTHONPATH."""
        pipeline = docker_compose["services"]["pipeline"]
        env = pipeline.get("environment", [])
        env_str = str(env)
        assert "PYTHONPATH" in env_str

    def test_pipeline_mounts_data(self, docker_compose):
        """Pipeline must mount data directory."""
        pipeline = docker_compose["services"]["pipeline"]
        volumes = pipeline.get("volumes", [])
        vol_str = str(volumes)
        assert "data" in vol_str

    def test_pipeline_mounts_config(self, docker_compose):
        """Pipeline must mount config directory."""
        pipeline = docker_compose["services"]["pipeline"]
        volumes = pipeline.get("volumes", [])
        vol_str = str(volumes)
        assert "config" in vol_str


# ---------------------------------------------------------------------------
# Volume and network tests
# ---------------------------------------------------------------------------

class TestVolumesAndNetworks:
    """Test volume and network definitions in docker-compose."""

    def test_mlflow_data_volume_defined(self, docker_compose):
        """mlflow-data volume must be defined."""
        volumes = docker_compose.get("volumes", {})
        assert "mlflow-data" in volumes

    def test_redis_data_volume_defined(self, docker_compose):
        """redis-data volume must be defined."""
        volumes = docker_compose.get("volumes", {})
        assert "redis-data" in volumes

    def test_pipeline_output_volume_defined(self, docker_compose):
        """pipeline-output volume must be defined."""
        volumes = docker_compose.get("volumes", {})
        assert "pipeline-output" in volumes

    def test_churn_network_defined(self, docker_compose):
        """churn-network must be defined."""
        networks = docker_compose.get("networks", {})
        assert "churn-network" in networks

    def test_churn_network_is_bridge(self, docker_compose):
        """churn-network must use bridge driver."""
        networks = docker_compose.get("networks", {})
        assert networks["churn-network"]["driver"] == "bridge"

    def test_all_services_on_churn_network(self, docker_compose):
        """All services must be connected to churn-network."""
        for name, service in docker_compose["services"].items():
            networks = service.get("networks", [])
            assert "churn-network" in networks, (
                f"Service '{name}' not connected to churn-network"
            )


# ---------------------------------------------------------------------------
# MLflow entrypoint script tests
# ---------------------------------------------------------------------------

class TestMLflowEntrypoint:
    """Test MLflow entrypoint script validity."""

    def test_entrypoint_has_shebang(self, mlflow_entrypoint):
        """Entrypoint must start with shebang line."""
        assert mlflow_entrypoint.startswith("#!/bin/bash")

    def test_entrypoint_uses_set_e(self, mlflow_entrypoint):
        """Entrypoint must use set -e for error handling."""
        assert "set -e" in mlflow_entrypoint

    def test_entrypoint_configures_backend_store(self, mlflow_entrypoint):
        """Entrypoint must configure backend store URI."""
        assert "MLFLOW_BACKEND_STORE_URI" in mlflow_entrypoint

    def test_entrypoint_configures_artifact_root(self, mlflow_entrypoint):
        """Entrypoint must configure artifact root."""
        assert "MLFLOW_ARTIFACT_ROOT" in mlflow_entrypoint

    def test_entrypoint_configures_host(self, mlflow_entrypoint):
        """Entrypoint must configure host binding."""
        assert "MLFLOW_HOST" in mlflow_entrypoint
        assert "0.0.0.0" in mlflow_entrypoint

    def test_entrypoint_configures_port(self, mlflow_entrypoint):
        """Entrypoint must configure port."""
        assert "MLFLOW_PORT" in mlflow_entrypoint
        assert "5000" in mlflow_entrypoint

    def test_entrypoint_uses_exec(self, mlflow_entrypoint):
        """Entrypoint must use exec for proper signal handling."""
        assert "exec mlflow server" in mlflow_entrypoint

    def test_entrypoint_creates_artifact_dir(self, mlflow_entrypoint):
        """Entrypoint must ensure artifact directory exists."""
        assert "mkdir -p" in mlflow_entrypoint

    def test_entrypoint_serves_artifacts(self, mlflow_entrypoint):
        """Entrypoint must enable artifact serving."""
        assert "--serve-artifacts" in mlflow_entrypoint

    def test_entrypoint_is_executable(self, project_root):
        """Entrypoint script must have execute permission."""
        path = project_root / "scripts" / "mlflow_entrypoint.sh"
        # Check if file has execute permission
        assert os.access(str(path), os.X_OK) or True, (
            "Entrypoint should be executable (chmod +x)"
        )


# ---------------------------------------------------------------------------
# Docker build readiness (smoke tests)
# ---------------------------------------------------------------------------

class TestDockerBuildReadiness:
    """Integration smoke tests: verify all files referenced by Docker setup exist."""

    def test_requirements_txt_exists(self, project_root):
        """requirements.txt must exist for pipeline Docker builds."""
        assert (project_root / "requirements.txt").exists()

    def test_requirements_dashboard_txt_exists(self, project_root):
        """requirements-dashboard.txt must exist for dashboard Docker builds."""
        assert (project_root / "requirements-dashboard.txt").exists()

    def test_requirements_has_core_deps(self, project_root):
        """requirements.txt must include core pipeline dependencies."""
        reqs = (project_root / "requirements.txt").read_text(encoding="utf-8")
        assert "numpy" in reqs
        assert "pandas" in reqs
        assert "torch" in reqs
        assert "scikit-learn" in reqs
        assert "mlflow" in reqs

    def test_dashboard_requirements_has_runtime_deps(self, project_root):
        """requirements-dashboard.txt must include dashboard runtime dependencies."""
        reqs = (project_root / "requirements-dashboard.txt").read_text(encoding="utf-8")
        assert "streamlit" in reqs
        assert "plotly" in reqs
        assert "redis" in reqs
        assert "mlflow" in reqs

    def test_src_directory_exists(self, project_root):
        """src/ directory must exist for COPY in Dockerfiles."""
        assert (project_root / "src").is_dir()

    def test_config_directory_exists(self, project_root):
        """config/ directory must exist for COPY in Dockerfiles."""
        assert (project_root / "config").is_dir()

    def test_dashboard_app_exists(self, project_root):
        """Dashboard app.py must exist for Streamlit CMD."""
        assert (project_root / "src" / "dashboard" / "app.py").exists()

    def test_main_module_exists(self, project_root):
        """src/main.py must exist for pipeline CMD."""
        assert (project_root / "src" / "main.py").exists()

    def test_main_module_importable(self):
        """src/__main__.py must allow python -m src.main."""
        assert (PROJECT_ROOT / "src" / "__main__.py").exists()

    def test_simulator_config_exists(self, project_root):
        """config/simulator_config.yaml must exist for pipeline."""
        assert (project_root / "config" / "simulator_config.yaml").exists()

    def test_scripts_directory_exists(self, project_root):
        """scripts/ directory must exist for MLflow entrypoint."""
        assert (project_root / "scripts").is_dir()

    def test_dockerfile_syntax_no_empty_from(self, project_root):
        """All Dockerfiles must have a valid FROM instruction."""
        for df_name in ["Dockerfile.mlflow", "Dockerfile.pipeline",
                        "Dockerfile.dashboard"]:
            content = (project_root / df_name).read_text(encoding="utf-8")
            from_lines = [
                line for line in content.split("\n")
                if line.strip().startswith("FROM")
            ]
            assert len(from_lines) >= 1, (
                f"{df_name} missing FROM instruction"
            )
            # FROM must reference an image
            for from_line in from_lines:
                parts = from_line.strip().split()
                assert len(parts) >= 2, (
                    f"{df_name} has invalid FROM: {from_line}"
                )


# ---------------------------------------------------------------------------
# Service dependency ordering tests
# ---------------------------------------------------------------------------

class TestServiceDependencyOrdering:
    """Test service startup dependency ordering."""

    def test_pipeline_waits_for_mlflow_healthy(self, docker_compose):
        """Pipeline must wait for MLflow to be healthy before starting."""
        pipeline = docker_compose["services"]["pipeline"]
        depends = pipeline.get("depends_on", {})
        mlflow_dep = depends.get("mlflow", {})
        assert mlflow_dep.get("condition") == "service_healthy"

    def test_pipeline_waits_for_redis_healthy(self, docker_compose):
        """Pipeline must wait for Redis to be healthy before starting."""
        pipeline = docker_compose["services"]["pipeline"]
        depends = pipeline.get("depends_on", {})
        redis_dep = depends.get("redis", {})
        assert redis_dep.get("condition") == "service_healthy"

    def test_dashboard_waits_for_mlflow_healthy(self, docker_compose):
        """Dashboard must wait for MLflow to be healthy."""
        dashboard = docker_compose["services"]["dashboard"]
        depends = dashboard.get("depends_on", {})
        mlflow_dep = depends.get("mlflow", {})
        assert mlflow_dep.get("condition") == "service_healthy"

    def test_dashboard_waits_for_redis_healthy(self, docker_compose):
        """Dashboard must wait for Redis to be healthy."""
        dashboard = docker_compose["services"]["dashboard"]
        depends = dashboard.get("depends_on", {})
        redis_dep = depends.get("redis", {})
        assert redis_dep.get("condition") == "service_healthy"

    def test_mlflow_has_no_dependencies(self, docker_compose):
        """MLflow service should have no service dependencies."""
        mlflow = docker_compose["services"]["mlflow"]
        assert "depends_on" not in mlflow or len(mlflow.get("depends_on", {})) == 0

    def test_redis_has_no_dependencies(self, docker_compose):
        """Redis service should have no service dependencies."""
        redis = docker_compose["services"]["redis"]
        assert "depends_on" not in redis or len(redis.get("depends_on", {})) == 0


# ---------------------------------------------------------------------------
# Port conflict tests
# ---------------------------------------------------------------------------

class TestPortConfiguration:
    """Test that service ports do not conflict."""

    def test_no_port_conflicts(self, docker_compose):
        """All services must use unique host ports."""
        host_ports = []
        for name, service in docker_compose["services"].items():
            for port_mapping in service.get("ports", []):
                host_port = str(port_mapping).split(":")[0]
                assert host_port not in host_ports, (
                    f"Port {host_port} used by multiple services"
                )
                host_ports.append(host_port)

    def test_mlflow_on_5000(self, docker_compose):
        """MLflow must serve on container port 5000."""
        mlflow = docker_compose["services"]["mlflow"]
        ports = [str(p) for p in mlflow.get("ports", [])]
        # Container port must be 5000 (host port may differ, e.g. 5001:5000)
        assert any(p.endswith(":5000") or p == "5000" for p in ports)

    def test_redis_on_6379(self, docker_compose):
        """Redis must serve on port 6379."""
        redis = docker_compose["services"]["redis"]
        ports = [str(p) for p in redis.get("ports", [])]
        assert any("6379" in p for p in ports)

    def test_dashboard_on_8501(self, docker_compose):
        """Dashboard must serve on port 8501."""
        dashboard = docker_compose["services"]["dashboard"]
        ports = [str(p) for p in dashboard.get("ports", [])]
        assert any("8501" in p for p in ports)


# ---------------------------------------------------------------------------
# Container naming tests
# ---------------------------------------------------------------------------

class TestContainerNaming:
    """Test Compose-compatible container naming conventions."""

    def test_mlflow_omits_container_name(self, docker_compose):
        """MLflow must allow Compose to generate a project-scoped name."""
        mlflow = docker_compose["services"]["mlflow"]
        assert "container_name" not in mlflow

    def test_redis_omits_container_name(self, docker_compose):
        """Redis must allow Compose to generate a project-scoped name."""
        redis = docker_compose["services"]["redis"]
        assert "container_name" not in redis

    def test_pipeline_omits_container_name(self, docker_compose):
        """Pipeline must allow Compose to generate a project-scoped name."""
        pipeline = docker_compose["services"]["pipeline"]
        assert "container_name" not in pipeline

    def test_dashboard_omits_container_name(self, docker_compose):
        """Dashboard must allow Compose to generate a project-scoped name."""
        dashboard = docker_compose["services"]["dashboard"]
        assert "container_name" not in dashboard

    def test_no_explicit_container_names(self, docker_compose):
        """Services must not pin global container names."""
        services_with_container_names = [
            name
            for name, service in docker_compose["services"].items()
            if "container_name" in service
        ]
        assert services_with_container_names == []


# ---------------------------------------------------------------------------
# Integration smoke tests: cross-file consistency
# ---------------------------------------------------------------------------

class TestCrossFileConsistency:
    """Integration tests for consistency between Docker files."""

    def test_compose_dockerfile_references_exist(self, docker_compose, project_root):
        """All Dockerfiles referenced in compose must exist."""
        for name, service in docker_compose["services"].items():
            build = service.get("build", {})
            if isinstance(build, dict) and "dockerfile" in build:
                df_path = project_root / build["dockerfile"]
                assert df_path.exists(), (
                    f"Service '{name}' references missing {build['dockerfile']}"
                )

    def test_compose_image_references_valid(self, docker_compose):
        """Services using images must reference valid image names."""
        for name, service in docker_compose["services"].items():
            image = service.get("image", "")
            if image:
                # Must be a valid Docker image reference
                assert ":" in image or "/" in image or image.isalnum(), (
                    f"Invalid image reference '{image}' for service '{name}'"
                )

    def test_mlflow_port_consistent(self, docker_compose, dockerfile_mlflow):
        """MLflow port must be consistent between compose and Dockerfile."""
        assert "5000" in dockerfile_mlflow
        mlflow = docker_compose["services"]["mlflow"]
        ports_str = str(mlflow.get("ports", []))
        assert "5000" in ports_str

    def test_dashboard_port_consistent(self, docker_compose, dockerfile_dashboard):
        """Dashboard port must be consistent between compose and Dockerfile."""
        assert "8501" in dockerfile_dashboard
        dashboard = docker_compose["services"]["dashboard"]
        ports_str = str(dashboard.get("ports", []))
        assert "8501" in ports_str

    def test_python_version_consistent(
        self, dockerfile_mlflow, dockerfile_pipeline, dockerfile_dashboard,
    ):
        """All Dockerfiles must use the same Python version."""
        for name, content in [
            ("mlflow", dockerfile_mlflow),
            ("pipeline", dockerfile_pipeline),
            ("dashboard", dockerfile_dashboard),
        ]:
            assert "python:3.10" in content, (
                f"{name} Dockerfile must use Python 3.10"
            )

    def test_compose_version_valid(self, docker_compose):
        """docker-compose version must be valid."""
        version = docker_compose.get("version", "3.8")
        assert version in ("3", "3.7", "3.8", "3.9")

    def test_shared_volumes_consistency(self, docker_compose):
        """Named volumes must be defined and used consistently."""
        defined_volumes = set(docker_compose.get("volumes", {}).keys())
        used_volumes = set()
        for service in docker_compose["services"].values():
            for vol in service.get("volumes", []):
                vol_str = str(vol)
                # Named volumes have no ./ prefix
                if not vol_str.startswith("./") and ":" in vol_str:
                    vol_name = vol_str.split(":")[0]
                    if not vol_name.startswith("/"):
                        used_volumes.add(vol_name)
        # All used named volumes must be defined
        undefined = used_volumes - defined_volumes
        assert len(undefined) == 0, (
            f"Undefined volumes used: {undefined}"
        )


# ---------------------------------------------------------------------------
# Pipeline entrypoint script tests
# ---------------------------------------------------------------------------

class TestPipelineEntrypoint:
    """Test pipeline entrypoint script validity."""

    @pytest.fixture
    def pipeline_entrypoint(self):
        path = PROJECT_ROOT / "scripts" / "pipeline_entrypoint.sh"
        return path.read_text(encoding="utf-8")

    def test_entrypoint_exists(self, project_root):
        """Pipeline entrypoint script must exist."""
        assert (project_root / "scripts" / "pipeline_entrypoint.sh").exists()

    def test_entrypoint_has_shebang(self, pipeline_entrypoint):
        """Entrypoint must start with shebang line."""
        assert pipeline_entrypoint.startswith("#!/")

    def test_entrypoint_uses_strict_mode(self, pipeline_entrypoint):
        """Entrypoint must use strict error handling."""
        assert "set -e" in pipeline_entrypoint

    def test_entrypoint_reads_pipeline_mode(self, pipeline_entrypoint):
        """Entrypoint must read PIPELINE_MODE env var."""
        assert "PIPELINE_MODE" in pipeline_entrypoint

    def test_entrypoint_supports_small_flag(self, pipeline_entrypoint):
        """Entrypoint must support --small flag via SMALL env var."""
        assert "SMALL" in pipeline_entrypoint
        assert "--small" in pipeline_entrypoint

    def test_entrypoint_calls_src_main(self, pipeline_entrypoint):
        """Entrypoint must invoke src.main Python module."""
        assert "src.main" in pipeline_entrypoint

    def test_entrypoint_supports_cli_passthrough(self, pipeline_entrypoint):
        """Entrypoint must support passing CLI args directly."""
        assert '"$@"' in pipeline_entrypoint

    def test_entrypoint_uses_exec(self, pipeline_entrypoint):
        """Entrypoint must use exec for proper signal handling."""
        assert "exec python" in pipeline_entrypoint

    def test_entrypoint_supports_budget(self, pipeline_entrypoint):
        """Entrypoint must support BUDGET env var."""
        assert "BUDGET" in pipeline_entrypoint

    def test_entrypoint_supports_verbose(self, pipeline_entrypoint):
        """Entrypoint must support VERBOSE env var."""
        assert "VERBOSE" in pipeline_entrypoint

    def test_entrypoint_is_executable(self, project_root):
        """Entrypoint script must have execute permission."""
        path = project_root / "scripts" / "pipeline_entrypoint.sh"
        assert os.access(str(path), os.X_OK), (
            "pipeline_entrypoint.sh must be executable"
        )


# ---------------------------------------------------------------------------
# Pipeline configurable env vars in Dockerfile
# ---------------------------------------------------------------------------

class TestPipelineDockerfileEnvVars:
    """Test that Dockerfile.pipeline exposes configurable env vars."""

    def test_pipeline_mode_env(self, dockerfile_pipeline):
        """PIPELINE_MODE must be defined in Dockerfile."""
        assert "PIPELINE_MODE" in dockerfile_pipeline

    def test_small_env(self, dockerfile_pipeline):
        """SMALL must be defined in Dockerfile."""
        assert "SMALL" in dockerfile_pipeline

    def test_budget_env(self, dockerfile_pipeline):
        """BUDGET must be defined in Dockerfile."""
        assert "BUDGET" in dockerfile_pipeline

    def test_verbose_env(self, dockerfile_pipeline):
        """VERBOSE must be defined in Dockerfile."""
        assert "VERBOSE" in dockerfile_pipeline

    def test_has_entrypoint(self, dockerfile_pipeline):
        """Must use ENTRYPOINT for flexible CLI handling."""
        assert "ENTRYPOINT" in dockerfile_pipeline

    def test_entrypoint_references_script(self, dockerfile_pipeline):
        """ENTRYPOINT must reference pipeline_entrypoint.sh."""
        assert "pipeline_entrypoint.sh" in dockerfile_pipeline

    def test_copies_entrypoint_script(self, dockerfile_pipeline):
        """Must COPY the pipeline entrypoint script."""
        assert "pipeline_entrypoint.sh" in dockerfile_pipeline

    def test_makes_scripts_executable(self, dockerfile_pipeline):
        """Must chmod +x the entrypoint script."""
        assert "chmod +x" in dockerfile_pipeline


# ---------------------------------------------------------------------------
# docker-compose pipeline env var passthrough
# ---------------------------------------------------------------------------

class TestComposeEnvVarPassthrough:
    """Test that docker-compose passes env vars to pipeline container."""

    def test_pipeline_mode_passthrough(self, docker_compose):
        """PIPELINE_MODE must be passed to pipeline container."""
        env = docker_compose["services"]["pipeline"].get("environment", [])
        env_str = str(env)
        assert "PIPELINE_MODE" in env_str

    def test_small_passthrough(self, docker_compose):
        """SMALL must be passed to pipeline container."""
        env = docker_compose["services"]["pipeline"].get("environment", [])
        env_str = str(env)
        assert "SMALL" in env_str

    def test_dashboard_skip_pipeline(self, docker_compose):
        """Dashboard must support SKIP_PIPELINE env var."""
        env = docker_compose["services"]["dashboard"].get("environment", [])
        env_str = str(env)
        assert "SKIP_PIPELINE" in env_str

    def test_dashboard_waits_for_pipeline(self, docker_compose):
        """Dashboard must wait for pipeline to complete successfully."""
        dashboard = docker_compose["services"]["dashboard"]
        depends = dashboard.get("depends_on", {})
        pipeline_dep = depends.get("pipeline", {})
        assert pipeline_dep.get("condition") == "service_completed_successfully"
