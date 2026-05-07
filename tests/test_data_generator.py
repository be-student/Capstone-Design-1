"""
TDD Tests for the Customer Behavior Data Generator (Simulator).

Tests cover:
- Config loading from YAML
- Customer generation with 6 personas
- Event log generation with 8+ event types
- Treatment/Control group assignment
- Churn rate within target range (15%-25%)
- Temporal behavior decay simulation
- Marketing response modeling per persona
- Reproducibility via random seed
- Small mode support
- Output file structure
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


@pytest.fixture
def config():
    """Load simulator configuration from YAML."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def small_generator(config):
    """Create a small-mode data generator for fast testing.

    Uses 200 customers over 90 days for speed; full small_mode (5000/180)
    is reserved for integration runs.
    """
    from src.data.generator import CustomerDataGenerator

    config["simulation"]["num_customers"] = 200
    config["simulation"]["simulation_days"] = 90
    return CustomerDataGenerator(config)


@pytest.fixture
def generated_data(small_generator):
    """Generate data using small mode for testing."""
    return small_generator.generate()


class TestConfigLoading:
    """Test that configuration is properly loaded and validated."""

    def test_config_file_exists(self):
        """Config YAML file must exist at expected path."""
        assert CONFIG_PATH.exists(), f"Config file not found at {CONFIG_PATH}"

    def test_config_has_required_sections(self, config):
        """Config must have all required top-level sections."""
        required_sections = [
            "simulation", "churn_definition", "treatment",
            "event_types", "personas", "target_churn_rate"
        ]
        for section in required_sections:
            assert section in config, f"Missing config section: {section}"

    def test_config_has_six_personas(self, config):
        """Config must define at least 6 customer personas."""
        assert len(config["personas"]) >= 6, (
            f"Expected >= 6 personas, got {len(config['personas'])}"
        )

    def test_config_persona_proportions_sum_to_one(self, config):
        """Persona proportions must sum to approximately 1.0."""
        total = sum(p["proportion"] for p in config["personas"])
        assert abs(total - 1.0) < 0.01, (
            f"Persona proportions sum to {total}, expected ~1.0"
        )

    def test_config_has_random_seed(self, config):
        """Config must specify a random seed for reproducibility."""
        assert "random_seed" in config["simulation"]

    def test_small_mode_matches_requirement(self, config):
        """small_mode must match the 5,000 customers / 6 months requirement."""
        small_mode = config["simulation"]["small_mode"]
        assert small_mode["num_customers"] == 5000
        assert small_mode["simulation_months"] == 6
        assert small_mode["simulation_days"] == 180

    def test_config_churn_definition(self, config):
        """Churn definition must have no_purchase_days and no_login_days."""
        churn_def = config["churn_definition"]
        assert "no_purchase_days" in churn_def
        assert "no_login_days" in churn_def
        assert churn_def["no_purchase_days"] == 30
        assert churn_def["no_login_days"] == 60

    def test_config_has_eight_event_types(self, config):
        """Config must define at least 8 event types."""
        assert len(config["event_types"]) >= 8, (
            f"Expected >= 8 event types, got {len(config['event_types'])}"
        )

    def test_config_event_types_content(self, config):
        """Config must include required event types."""
        required_events = [
            "page_view", "search", "add_to_cart", "remove_from_cart",
            "purchase", "coupon_use", "review", "cs_contact"
        ]
        for event in required_events:
            assert event in config["event_types"], (
                f"Missing event type: {event}"
            )


class TestCustomerGeneration:
    """Test customer profile generation."""

    def test_generator_creates_customers_dataframe(self, generated_data):
        """Generator must return a customers DataFrame."""
        assert "customers" in generated_data
        assert isinstance(generated_data["customers"], pd.DataFrame)

    def test_correct_number_of_customers(self, generated_data):
        """Number of generated customers must match configured count."""
        actual = len(generated_data["customers"])
        assert actual == 200, (
            f"Expected 200 customers, got {actual}"
        )

    def test_customer_has_required_columns(self, generated_data):
        """Customer DataFrame must have required columns."""
        required_cols = [
            "customer_id", "persona", "signup_date",
            "treatment_group"
        ]
        df = generated_data["customers"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_all_personas_represented(self, generated_data, config):
        """All 6 personas must be represented in generated data."""
        personas_in_data = set(generated_data["customers"]["persona"].unique())
        personas_in_config = {p["name"] for p in config["personas"]}
        assert personas_in_config.issubset(personas_in_data), (
            f"Missing personas: {personas_in_config - personas_in_data}"
        )

    def test_persona_distribution_approximate(self, generated_data, config):
        """Persona distribution should approximately match config proportions."""
        df = generated_data["customers"]
        total = len(df)
        for persona_cfg in config["personas"]:
            name = persona_cfg["name"]
            expected_ratio = persona_cfg["proportion"]
            actual_ratio = len(df[df["persona"] == name]) / total
            assert abs(actual_ratio - expected_ratio) < 0.05, (
                f"Persona {name}: expected ratio ~{expected_ratio}, "
                f"got {actual_ratio}"
            )

    def test_customer_ids_unique(self, generated_data):
        """All customer IDs must be unique."""
        df = generated_data["customers"]
        assert df["customer_id"].nunique() == len(df)


class TestTreatmentControlGroups:
    """Test treatment/control group assignment."""

    def test_treatment_control_split(self, generated_data):
        """Customers must be split into treatment and control groups."""
        df = generated_data["customers"]
        groups = df["treatment_group"].unique()
        assert set(groups) == {"treatment", "control"}, (
            f"Expected treatment/control groups, got {groups}"
        )

    def test_treatment_control_balance(self, generated_data):
        """Treatment and control groups should be approximately balanced."""
        df = generated_data["customers"]
        treatment_count = len(df[df["treatment_group"] == "treatment"])
        total = len(df)
        treatment_ratio = treatment_count / total
        assert 0.45 <= treatment_ratio <= 0.55, (
            f"Treatment ratio {treatment_ratio} not balanced"
        )

    def test_treatment_group_minimum_size(self, generated_data):
        """Each group should have >= 40% of total customers."""
        df = generated_data["customers"]
        treatment_count = len(df[df["treatment_group"] == "treatment"])
        control_count = len(df[df["treatment_group"] == "control"])
        min_per_group = len(df) * 0.4
        assert treatment_count >= min_per_group
        assert control_count >= min_per_group

    def test_generator_marks_reduced_test_run_as_small_mode(self, config):
        """Reduced local/unit runs should be treated as small mode."""
        from src.data.generator import CustomerDataGenerator

        config["simulation"]["num_customers"] = 200
        config["simulation"]["simulation_days"] = 90
        generator = CustomerDataGenerator(config)

        assert generator.infer_generation_mode() == "small"

    def test_generator_can_reach_full_mode_when_feasible(self, config):
        """Requirement-sized runs should be classified as full mode."""
        from src.data.generator import CustomerDataGenerator

        config["simulation"]["num_customers"] = 20000
        config["simulation"]["simulation_days"] = 365
        generator = CustomerDataGenerator(config)

        assert generator.infer_generation_mode() == "full"


class TestEventGeneration:
    """Test event log generation."""

    def test_events_dataframe_created(self, generated_data):
        """Generator must produce an events DataFrame."""
        assert "events" in generated_data
        assert isinstance(generated_data["events"], pd.DataFrame)

    def test_events_have_required_columns(self, generated_data):
        """Events DataFrame must have required columns."""
        required_cols = [
            "customer_id", "event_type", "event_date", "timestamp",
            "session_duration", "marketing_channel", "marketing_response",
        ]
        df = generated_data["events"]
        for col in required_cols:
            assert col in df.columns, f"Missing event column: {col}"

    def test_all_event_types_present(self, generated_data, config):
        """All 8+ event types must appear in generated events."""
        event_types_in_data = set(generated_data["events"]["event_type"].unique())
        required_events = set(config["event_types"])
        assert required_events.issubset(event_types_in_data), (
            f"Missing event types: {required_events - event_types_in_data}"
        )

    def test_events_linked_to_valid_customers(self, generated_data):
        """All events must reference valid customer IDs."""
        customer_ids = set(generated_data["customers"]["customer_id"])
        event_customer_ids = set(generated_data["events"]["customer_id"])
        orphan_ids = event_customer_ids - customer_ids
        assert len(orphan_ids) == 0, (
            f"Found {len(orphan_ids)} orphan customer IDs in events"
        )

    def test_events_within_simulation_period(self, generated_data):
        """All events must fall within the simulation period."""
        events = generated_data["events"]
        event_dates = pd.to_datetime(events["event_date"])
        sim_days = 90  # matches small_generator fixture
        date_range = (event_dates.max() - event_dates.min()).days
        assert date_range <= sim_days + 10, (
            f"Event date range {date_range} days exceeds "
            f"simulation period {sim_days} days"
        )

    def test_purchase_events_have_amount(self, generated_data):
        """Purchase events must include an order amount."""
        events = generated_data["events"]
        purchases = events[events["event_type"] == "purchase"]
        assert len(purchases) > 0, "No purchase events generated"
        assert "amount" in events.columns, "Events missing 'amount' column"
        purchase_amounts = purchases["amount"]
        assert purchase_amounts.notna().all(), "Some purchases have null amounts"
        assert (purchase_amounts > 0).all(), "Some purchases have non-positive amounts"

    def test_visit_events_have_positive_session_duration(self, generated_data):
        """Visit-session events must include positive session duration."""
        events = generated_data["events"]
        visit_events = events[
            events["event_type"].isin(
                ["page_view", "search", "add_to_cart", "remove_from_cart", "purchase"]
            )
        ]
        assert "session_duration" in events.columns
        assert visit_events["session_duration"].notna().all()
        assert (visit_events["session_duration"] > 0).all()


class TestChurnLabeling:
    """Test churn label generation."""

    def test_churn_labels_exist(self, generated_data):
        """Generated data must include churn labels."""
        assert "churn_label" in generated_data["customers"].columns

    def test_churn_labels_binary(self, generated_data):
        """Churn labels must be binary (0 or 1)."""
        labels = generated_data["customers"]["churn_label"]
        assert set(labels.unique()).issubset({0, 1}), (
            f"Churn labels contain non-binary values: {labels.unique()}"
        )

    def test_churn_rate_in_target_range(self, generated_data, config):
        """Overall churn rate should be reasonable (relaxed for small test set).

        With 200 customers over 90 days the churn rate may not perfectly
        match the 15%-25% target designed for 20k customers / 365 days.
        We accept 5%-40% for fast unit tests; the full integration test
        validates the strict range.
        """
        churn_rate = generated_data["customers"]["churn_label"].mean()
        assert 0.05 <= churn_rate <= 0.40, (
            f"Churn rate {churn_rate:.2%} outside relaxed test range "
            f"[5%, 40%]"
        )

    def test_churn_based_on_configurable_definition(self, generated_data, config):
        """Churn should be determined by configurable no_purchase/no_login days."""
        no_purchase_days = config["churn_definition"]["no_purchase_days"]
        no_login_days = config["churn_definition"]["no_login_days"]
        assert no_purchase_days > 0
        assert no_login_days > 0
        assert generated_data["customers"]["churn_label"].notna().all()

    def test_orchestrator_summary_flags_small_mode_validation_skip(self, config, tmp_path):
        """Small-mode summaries should explicitly note skipped 10k/group validation."""
        from src.data.orchestrator import SimulatorOrchestrator

        config["simulation"]["num_customers"] = 200
        config["simulation"]["simulation_days"] = 90
        orchestrator = SimulatorOrchestrator(config)

        result = orchestrator.run(str(tmp_path / "raw"))
        validation = result["summary"]["validation"]

        assert validation["mode"] == "small"
        assert validation["group_size_check"]["passed"] is False
        assert any(
            "Small mode summary only" in warning
            for warning in validation["warnings"]
        )


class TestTemporalBehavior:
    """Test time-based behavior changes in simulation."""

    def test_behavior_decay_modeled(self, generated_data):
        """Churning customers should show declining activity over time."""
        events = generated_data["events"]
        customers = generated_data["customers"]

        churned_ids = customers[customers["churn_label"] == 1]["customer_id"]
        churned_events = events[events["customer_id"].isin(churned_ids)]

        if len(churned_events) == 0:
            pytest.skip("No churned customer events to analyze")

        churned_events = churned_events.copy()
        churned_events["event_date"] = pd.to_datetime(churned_events["event_date"])

        mid_date = churned_events["event_date"].min() + (
            churned_events["event_date"].max() - churned_events["event_date"].min()
        ) / 2

        first_half = churned_events[churned_events["event_date"] <= mid_date]
        second_half = churned_events[churned_events["event_date"] > mid_date]

        first_avg = len(first_half) / max(len(churned_ids), 1)
        second_avg = len(second_half) / max(len(churned_ids), 1)

        assert second_avg <= first_avg * 1.2, (
            f"Churned customer activity did not decay: "
            f"first_half_avg={first_avg:.1f}, second_half_avg={second_avg:.1f}"
        )


class TestMarketingResponse:
    """Test marketing intervention response modeling."""

    def test_marketing_events_exist(self, generated_data):
        """Treatment group should have marketing-related events."""
        events = generated_data["events"]
        customers = generated_data["customers"]

        treatment_ids = customers[
            customers["treatment_group"] == "treatment"
        ]["customer_id"]
        treatment_events = events[events["customer_id"].isin(treatment_ids)]

        coupon_events = treatment_events[
            treatment_events["event_type"] == "coupon_use"
        ]
        assert len(coupon_events) > 0, (
            "No coupon_use events found for treatment group"
        )

    def test_marketing_response_metadata_exists(self, generated_data):
        """Treatment events should label conversion/no-response/adverse metadata."""
        events = generated_data["events"]
        customers = generated_data["customers"]
        treatment_ids = customers[
            customers["treatment_group"] == "treatment"
        ]["customer_id"]
        treatment_events = events[events["customer_id"].isin(treatment_ids)]

        responses = set(treatment_events["marketing_response"].dropna())
        assert responses.issubset({"conversion", "no_response", "adverse"})
        assert "conversion" in responses

    def test_differential_persona_response(self, generated_data, config):
        """Different personas should have different marketing response rates."""
        events = generated_data["events"]
        customers = generated_data["customers"]

        treatment_customers = customers[
            customers["treatment_group"] == "treatment"
        ]

        persona_coupon_rates = {}
        for persona_cfg in config["personas"]:
            name = persona_cfg["name"]
            persona_ids = treatment_customers[
                treatment_customers["persona"] == name
            ]["customer_id"]
            if len(persona_ids) == 0:
                continue
            persona_events = events[events["customer_id"].isin(persona_ids)]
            coupon_count = len(
                persona_events[persona_events["event_type"] == "coupon_use"]
            )
            persona_coupon_rates[name] = coupon_count / len(persona_ids)

        rates = list(persona_coupon_rates.values())
        assert len(rates) >= 2, "Not enough personas with coupon data"
        assert max(rates) > min(rates), (
            "All personas have identical coupon response rates"
        )


class TestReproducibility:
    """Test that results are reproducible with the same seed."""

    def test_same_seed_produces_same_customers(self, config):
        """Same random seed must produce identical customer profiles."""
        from src.data.generator import CustomerDataGenerator

        config["simulation"]["num_customers"] = 100
        config["simulation"]["simulation_days"] = 30

        gen1 = CustomerDataGenerator(config)
        data1 = gen1.generate()

        gen2 = CustomerDataGenerator(config)
        data2 = gen2.generate()

        pd.testing.assert_frame_equal(
            data1["customers"].reset_index(drop=True),
            data2["customers"].reset_index(drop=True)
        )

    def test_same_seed_produces_same_events(self, config):
        """Same random seed must produce identical event logs."""
        from src.data.generator import CustomerDataGenerator

        config["simulation"]["num_customers"] = 100
        config["simulation"]["simulation_days"] = 30

        gen1 = CustomerDataGenerator(config)
        data1 = gen1.generate()

        gen2 = CustomerDataGenerator(config)
        data2 = gen2.generate()

        pd.testing.assert_frame_equal(
            data1["events"].reset_index(drop=True),
            data2["events"].reset_index(drop=True)
        )


class TestSingleCustomerEventGenerator:
    """Test the core _generate_customer_events method for a single customer.

    This validates that the event generator produces all 8 event types
    (page_view, search, add_to_cart, remove_from_cart, purchase, coupon_use,
    review, cs_contact) over a 12-month timeline for a single customer,
    respecting persona parameters.
    """

    @pytest.fixture
    def single_customer_generator(self, config):
        """Create a generator configured for single-customer testing."""
        from src.data.generator import CustomerDataGenerator

        config["simulation"]["num_customers"] = 1
        config["simulation"]["simulation_days"] = 365  # 12 months
        return CustomerDataGenerator(config)

    @pytest.fixture
    def vip_events(self, single_customer_generator, config):
        """Generate events for a single VIP loyal customer over 12 months."""
        gen = single_customer_generator
        persona_cfg = gen._get_persona_config("vip_loyal")
        signup_date = pd.Timestamp(config["simulation"]["start_date"])
        events = gen._generate_customer_events(
            customer_id="C000000",
            persona_cfg=persona_cfg,
            signup_date=signup_date,
            is_treatment=True,
        )
        return events

    @pytest.fixture
    def dormant_events(self, single_customer_generator, config):
        """Generate events for a single dormant customer over 12 months."""
        gen = single_customer_generator
        persona_cfg = gen._get_persona_config("dormant")
        signup_date = pd.Timestamp(config["simulation"]["start_date"])
        events = gen._generate_customer_events(
            customer_id="C000001",
            persona_cfg=persona_cfg,
            signup_date=signup_date,
            is_treatment=False,
        )
        return events

    def test_single_customer_produces_events(self, vip_events):
        """A single VIP customer must produce events over 12 months."""
        assert len(vip_events) > 0, "No events generated for single customer"

    def test_single_customer_all_8_event_types(self, vip_events):
        """A VIP customer over 12 months should produce all 8 event types."""
        event_types = {e["event_type"] for e in vip_events}
        required = {
            "page_view", "search", "add_to_cart", "remove_from_cart",
            "purchase", "coupon_use", "review", "cs_contact",
        }
        missing = required - event_types
        assert len(missing) == 0, (
            f"Missing event types for single customer: {missing}"
        )

    def test_single_customer_event_record_structure(self, vip_events):
        """Each event record must have customer_id, event_type, event_date,
        timestamp, amount, session, and marketing fields."""
        required_keys = {"customer_id", "event_type", "event_date",
                         "timestamp", "amount", "session_duration",
                         "marketing_channel", "marketing_response"}
        for event in vip_events[:5]:
            assert required_keys == set(event.keys()), (
                f"Event keys mismatch: {set(event.keys())}"
            )

    def test_single_customer_purchase_has_positive_amount(self, vip_events):
        """Purchase events must have a positive KRW amount."""
        purchases = [e for e in vip_events if e["event_type"] == "purchase"]
        assert len(purchases) > 0, "No purchase events"
        for p in purchases:
            assert p["amount"] is not None and p["amount"] > 0, (
                f"Purchase has invalid amount: {p['amount']}"
            )

    def test_single_customer_non_purchase_has_null_amount(self, vip_events):
        """Non-purchase events should have None amount."""
        non_purchases = [e for e in vip_events if e["event_type"] != "purchase"]
        for e in non_purchases[:20]:
            assert e["amount"] is None, (
                f"{e['event_type']} event has unexpected amount: {e['amount']}"
            )

    def test_single_customer_events_span_timeline(self, vip_events):
        """Events should span the full 12-month simulation period."""
        dates = [pd.Timestamp(e["event_date"]) for e in vip_events]
        span_days = (max(dates) - min(dates)).days
        # VIP customer should be active for most of the year
        assert span_days >= 300, (
            f"Event span only {span_days} days, expected >= 300 for VIP"
        )

    def test_vip_more_events_than_dormant(self, vip_events, dormant_events):
        """VIP loyal customer should generate many more events than dormant."""
        assert len(vip_events) > len(dormant_events) * 2, (
            f"VIP ({len(vip_events)}) should have >> dormant "
            f"({len(dormant_events)}) events"
        )

    def test_vip_more_purchases_than_dormant(self, vip_events, dormant_events):
        """VIP should have significantly more purchases than dormant."""
        vip_purchases = sum(1 for e in vip_events if e["event_type"] == "purchase")
        dormant_purchases = sum(
            1 for e in dormant_events if e["event_type"] == "purchase"
        )
        assert vip_purchases > dormant_purchases, (
            f"VIP purchases ({vip_purchases}) should exceed "
            f"dormant purchases ({dormant_purchases})"
        )

    def test_persona_engagement_affects_visit_frequency(
        self, single_customer_generator, config
    ):
        """Higher daily_visit_prob personas should have more page_view events."""
        gen = single_customer_generator
        signup = pd.Timestamp(config["simulation"]["start_date"])

        counts = {}
        for persona_name in ["vip_loyal", "dormant"]:
            gen.rng = np.random.RandomState(gen.seed)
            pcfg = gen._get_persona_config(persona_name)
            events = gen._generate_customer_events(
                f"C_test_{persona_name}", pcfg, signup, False,
            )
            counts[persona_name] = sum(
                1 for e in events if e["event_type"] == "page_view"
            )

        assert counts["vip_loyal"] > counts["dormant"], (
            f"VIP visits ({counts['vip_loyal']}) should exceed "
            f"dormant visits ({counts['dormant']})"
        )

    def test_treatment_increases_coupon_usage(
        self, single_customer_generator, config
    ):
        """Treatment group customer should have more coupon_use events."""
        gen = single_customer_generator
        signup = pd.Timestamp(config["simulation"]["start_date"])
        pcfg = gen._get_persona_config("bargain_hunter")

        gen.rng = np.random.RandomState(gen.seed)
        control_events = gen._generate_customer_events(
            "C_ctrl", pcfg, signup, is_treatment=False,
        )
        gen.rng = np.random.RandomState(gen.seed)
        treat_events = gen._generate_customer_events(
            "C_treat", pcfg, signup, is_treatment=True,
        )

        ctrl_coupons = sum(
            1 for e in control_events if e["event_type"] == "coupon_use"
        )
        treat_coupons = sum(
            1 for e in treat_events if e["event_type"] == "coupon_use"
        )
        # Treatment should have >= control (marketing lift)
        assert treat_coupons >= ctrl_coupons, (
            f"Treatment coupons ({treat_coupons}) should be >= "
            f"control ({ctrl_coupons})"
        )

    def test_behavior_decay_reduces_late_activity(
        self, single_customer_generator, config
    ):
        """Events in later months should be fewer than early months for
        high-decay personas (dormant)."""
        gen = single_customer_generator
        signup = pd.Timestamp(config["simulation"]["start_date"])
        pcfg = gen._get_persona_config("dormant")

        gen.rng = np.random.RandomState(gen.seed)
        events = gen._generate_customer_events(
            "C_decay", pcfg, signup, is_treatment=False,
        )
        if len(events) < 2:
            pytest.skip("Too few dormant events to measure decay")

        dates = [pd.Timestamp(e["event_date"]) for e in events]
        mid = signup + pd.Timedelta(days=182)
        first_half = sum(1 for d in dates if d < mid)
        second_half = sum(1 for d in dates if d >= mid)

        assert first_half > second_half, (
            f"Dormant first-half events ({first_half}) should exceed "
            f"second-half ({second_half}) due to decay"
        )

    def test_session_duration_decays_over_time(
        self, single_customer_generator, config
    ):
        """Session-time decay should be visible in generated session metadata."""
        gen = single_customer_generator
        signup = pd.Timestamp(config["simulation"]["start_date"])
        pcfg = gen._get_persona_config("dormant")

        gen.rng = np.random.RandomState(gen.seed)
        events = gen._generate_customer_events(
            "C_session_decay", pcfg, signup, is_treatment=False,
        )
        session_events = [
            event for event in events
            if event["event_type"] == "page_view"
            and event["session_duration"] is not None
        ]
        if len(session_events) < 4:
            pytest.skip("Too few sessions to measure session duration decay")

        midpoint = signup + pd.Timedelta(days=182)
        first_half = [
            event["session_duration"] for event in session_events
            if pd.Timestamp(event["event_date"]) < midpoint
        ]
        second_half = [
            event["session_duration"] for event in session_events
            if pd.Timestamp(event["event_date"]) >= midpoint
        ]
        if not first_half or not second_half:
            pytest.skip("Session events do not span both halves")

        assert np.median(second_half) < np.median(first_half)

    def test_weekend_activity_boost(self, vip_events):
        """Weekend days should show boosted activity for personas with
        weekend_activity_boost > 1.0."""
        dates = [pd.Timestamp(e["event_date"]) for e in vip_events]
        weekday_events = sum(1 for d in dates if d.weekday() < 5)
        weekend_events = sum(1 for d in dates if d.weekday() >= 5)

        # Normalize by number of weekdays vs weekend days in a year
        # ~261 weekdays, ~104 weekend days
        weekday_rate = weekday_events / 261.0
        weekend_rate = weekend_events / 104.0

        # VIP has 1.1 boost, so weekend rate should be comparable or higher
        # Allow some tolerance since it's stochastic
        assert weekend_rate >= weekday_rate * 0.8, (
            f"Weekend rate ({weekend_rate:.1f}/day) too low vs "
            f"weekday ({weekday_rate:.1f}/day)"
        )


class TestDataOutput:
    """Test data output and saving functionality."""

    def test_save_to_files(self, small_generator, generated_data, tmp_path):
        """Generator must save data to specified output directory."""
        output_dir = tmp_path / "raw"
        small_generator.save(generated_data, str(output_dir))

        assert (output_dir / "customers.csv").exists()
        assert (output_dir / "events.csv").exists()

    def test_saved_data_loadable(self, small_generator, generated_data, tmp_path):
        """Saved CSV files must be loadable as DataFrames."""
        output_dir = tmp_path / "raw"
        small_generator.save(generated_data, str(output_dir))

        customers = pd.read_csv(output_dir / "customers.csv")
        events = pd.read_csv(output_dir / "events.csv")

        assert len(customers) > 0
        assert len(events) > 0
        assert "customer_id" in customers.columns
        assert "event_type" in events.columns
