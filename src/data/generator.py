"""
Customer Behavior Data Generator for E-Commerce Churn Prediction.

Generates simulated customer profiles and behavioral event logs based on
configurable persona parameters. Produces 8 event types (page_view, search,
add_to_cart, remove_from_cart, purchase, coupon_use, review, cs_contact)
over a configurable timeline with realistic behavior decay and marketing
response modeling.

Usage:
    generator = CustomerDataGenerator(config)
    data = generator.generate()
    generator.save(data, "data/raw")
"""

import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class CustomerDataGenerator:
    """Generate simulated customer profiles and event logs.

    Attributes:
        config: Full configuration dictionary loaded from YAML.
        rng: NumPy random generator seeded for reproducibility.
        num_customers: Number of customers to generate.
        simulation_days: Duration of simulation in days.
        start_date: Simulation start date.
        personas: List of persona configuration dicts.
        event_types: List of event type strings.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the generator with configuration.

        Args:
            config: Configuration dictionary (from simulator_config.yaml).
        """
        self.config = config
        sim_cfg = config["simulation"]
        self.seed = sim_cfg["random_seed"]
        self.rng = np.random.RandomState(self.seed)
        self.num_customers = sim_cfg["num_customers"]
        self.simulation_days = sim_cfg.get(
            "simulation_days",
            sim_cfg.get("simulation_months", 12) * 30
        )
        self.start_date = pd.Timestamp(sim_cfg["start_date"])
        self.end_date = self.start_date + timedelta(days=self.simulation_days - 1)

        self.personas = config["personas"]  # list of persona dicts
        self.event_types = config["event_types"]
        self.small_mode_cfg = sim_cfg.get("small_mode", {})

        churn_cfg = config["churn_definition"]
        self.no_purchase_days = churn_cfg["no_purchase_days"]
        self.no_login_days = churn_cfg["no_login_days"]
        self.churn_operator = churn_cfg.get("operator", "OR")

        treatment_cfg = config["treatment"]
        self.treatment_ratio = treatment_cfg["treatment_ratio"]
        self.min_group_size = int(treatment_cfg.get("min_group_size", 0))

        self.target_churn_min = config["target_churn_rate"]["min"]
        self.target_churn_max = config["target_churn_rate"]["max"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> Dict[str, pd.DataFrame]:
        """Generate customer profiles and event logs.

        Returns:
            Dictionary with 'customers' and 'events' DataFrames.
        """
        # Reset RNG for reproducibility on each call
        self.rng = np.random.RandomState(self.seed)

        customers_df = self._generate_customers()
        events_df = self._generate_all_events(customers_df)
        customers_df = self._label_churn(customers_df, events_df)
        customers_df, events_df = self._calibrate_full_mode_churn(
            customers_df,
            events_df,
        )
        self._warn_if_target_churn_out_of_range(customers_df)

        return {"customers": customers_df, "events": events_df}

    def save(
        self,
        data: Dict[str, pd.DataFrame],
        output_dir: str,
    ) -> None:
        """Save generated data to CSV files.

        Args:
            data: Dictionary with 'customers' and 'events' DataFrames.
            output_dir: Directory path to write CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)
        data["customers"].to_csv(
            os.path.join(output_dir, "customers.csv"), index=False
        )
        data["events"].to_csv(
            os.path.join(output_dir, "events.csv"), index=False
        )

    # ------------------------------------------------------------------
    # Customer Profile Generation
    # ------------------------------------------------------------------

    def _generate_customers(self) -> pd.DataFrame:
        """Generate customer profiles with persona and treatment assignment."""
        records: List[Dict[str, Any]] = []

        # Build persona assignment array based on proportions
        persona_names = [p["name"] for p in self.personas]
        proportions = [p["proportion"] for p in self.personas]

        # Assign personas to customers
        assignments = self.rng.choice(
            persona_names,
            size=self.num_customers,
            p=proportions,
        )

        # Assign treatment/control groups with deterministic sizing.
        treatment_flags = self._assign_treatment_flags()

        for i in range(self.num_customers):
            # Signup date: uniformly distributed in first 30% of simulation
            signup_offset = self.rng.randint(
                0, max(1, int(self.simulation_days * 0.3))
            )
            signup_date = self.start_date + timedelta(days=int(signup_offset))

            records.append({
                "customer_id": f"C{i:06d}",
                "persona": assignments[i],
                "signup_date": signup_date.strftime("%Y-%m-%d"),
                "treatment_group": "treatment" if treatment_flags[i] else "control",
            })

        return pd.DataFrame(records)

    def _assign_treatment_flags(self) -> np.ndarray:
        """Assign treatment/control flags while honoring feasible minimum sizes."""
        desired_treatment = int(round(self.num_customers * self.treatment_ratio))
        desired_treatment = max(0, min(self.num_customers, desired_treatment))

        if self.num_customers >= 2 * self.min_group_size and self.min_group_size > 0:
            desired_treatment = max(self.min_group_size, desired_treatment)
            desired_treatment = min(
                self.num_customers - self.min_group_size,
                desired_treatment,
            )

        treatment_flags = np.array(
            [True] * desired_treatment
            + [False] * (self.num_customers - desired_treatment),
            dtype=bool,
        )
        self.rng.shuffle(treatment_flags)
        return treatment_flags

    def infer_generation_mode(self) -> str:
        """Infer whether current settings should be treated as small mode."""
        small_customers = int(self.small_mode_cfg.get("num_customers", 0))
        small_days = int(self.small_mode_cfg.get("simulation_days", 0))

        if self.num_customers < 2 * self.min_group_size:
            return "small"
        if small_customers and self.num_customers <= small_customers:
            return "small"
        if small_days and self.simulation_days <= small_days:
            return "small"
        return "full"

    def _warn_if_target_churn_out_of_range(self, customers_df: pd.DataFrame) -> None:
        """Emit a warning when generated churn is outside the configured range."""
        churn_rate = float(customers_df["churn_label"].mean())
        if self.target_churn_min <= churn_rate <= self.target_churn_max:
            return

        warnings.warn(
            (
                f"Generated churn rate {churn_rate:.2%} is outside target range "
                f"[{self.target_churn_min:.0%}, {self.target_churn_max:.0%}] "
                f"for {self.infer_generation_mode()} mode."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    def _calibrate_full_mode_churn(
        self,
        customers_df: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Add deterministic reactivation events when full-mode churn is high.

        The churn definition remains the configured no-purchase/no-login rule.
        Calibration changes only the simulated behavior log by adding late
        page-view and purchase events for a small sample of churned customers,
        then recomputes labels from the same rule.
        """
        if self.infer_generation_mode() != "full" or events_df.empty:
            return customers_df, events_df

        churn_rate = float(customers_df["churn_label"].mean())
        if churn_rate <= self.target_churn_max:
            return customers_df, events_df

        target_rate = (self.target_churn_min + self.target_churn_max) / 2.0
        reactivate_count = int(
            np.ceil((churn_rate - target_rate) * len(customers_df))
        )
        churned = customers_df.loc[
            customers_df["churn_label"] == 1,
            ["customer_id", "persona"],
        ]
        if reactivate_count <= 0 or churned.empty:
            return customers_df, events_df

        selected = churned.sample(
            n=min(reactivate_count, len(churned)),
            random_state=self.seed,
        )
        persona_order_value = {
            persona["name"]: float(persona["avg_order_value"])
            for persona in self.personas
        }
        event_date = self.end_date - timedelta(days=7)
        rows: List[Dict[str, Any]] = []
        for _, row in selected.iterrows():
            amount = round(
                max(
                    1000.0,
                    persona_order_value.get(row["persona"], 50_000.0) * 0.8,
                ),
                -2,
            )
            timestamp = event_date.replace(hour=10, minute=0, second=0)
            rows.append(
                self._make_event(
                    row["customer_id"],
                    "page_view",
                    event_date,
                    timestamp,
                )
            )
            rows.append(
                self._make_event(
                    row["customer_id"],
                    "purchase",
                    event_date,
                    timestamp,
                    amount=amount,
                )
            )

        calibrated_events = pd.concat(
            [events_df, pd.DataFrame(rows)],
            ignore_index=True,
        )
        calibrated_customers = self._label_churn(customers_df, calibrated_events)
        return calibrated_customers, calibrated_events

    # ------------------------------------------------------------------
    # Event Generation (core behavior simulator)
    # ------------------------------------------------------------------

    def _get_persona_config(self, persona_name: str) -> Dict[str, Any]:
        """Look up persona config by name.

        Args:
            persona_name: The persona name string.

        Returns:
            Persona configuration dictionary.
        """
        for p in self.personas:
            if p["name"] == persona_name:
                return p
        raise ValueError(f"Unknown persona: {persona_name}")

    def _generate_all_events(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate events for all customers.

        Args:
            customers_df: DataFrame of customer profiles.

        Returns:
            DataFrame of all events across all customers.
        """
        all_events: List[Dict[str, Any]] = []

        for _, customer in customers_df.iterrows():
            persona_cfg = self._get_persona_config(customer["persona"])
            events = self._generate_customer_events(
                customer_id=customer["customer_id"],
                persona_cfg=persona_cfg,
                signup_date=pd.Timestamp(customer["signup_date"]),
                is_treatment=customer["treatment_group"] == "treatment",
            )
            all_events.extend(events)

        events_df = pd.DataFrame(all_events)
        if len(events_df) == 0:
            events_df = pd.DataFrame(columns=[
                "customer_id", "event_type", "event_date",
                "timestamp", "amount",
            ])

        return events_df.sort_values(
            ["customer_id", "timestamp"]
        ).reset_index(drop=True)

    def _generate_customer_events(
        self,
        customer_id: str,
        persona_cfg: Dict[str, Any],
        signup_date: pd.Timestamp,
        is_treatment: bool,
    ) -> List[Dict[str, Any]]:
        """Generate behavioral events for a single customer over the timeline.

        Produces 8 event types based on persona engagement parameters:
        - page_view: Daily visit probability
        - search: Searches per visit session
        - add_to_cart: Cart additions per visit
        - remove_from_cart: Partial cart removals
        - purchase: Based on cart-to-purchase rate and monthly frequency
        - coupon_use: Coupon redemption (boosted for treatment group)
        - review: Post-purchase review probability
        - cs_contact: Customer service contacts

        Behavior decays over time based on persona decay parameters.
        Treatment group gets marketing response lifts.
        Customers may explicitly churn (stop all activity) based on persona
        churn_probability evaluated monthly.

        Args:
            customer_id: Unique customer identifier.
            persona_cfg: Persona configuration dictionary.
            signup_date: Customer signup date.
            is_treatment: Whether customer is in treatment group.

        Returns:
            List of event dictionaries.
        """
        events: List[Dict[str, Any]] = []
        eng = persona_cfg["engagement"]
        decay = persona_cfg["behavior_decay"]
        mktg = persona_cfg["marketing_response"]

        # Base parameters
        daily_visit_prob = eng["daily_visit_prob"]
        search_per_visit = eng["search_per_visit"]
        cart_add_per_visit = eng["cart_add_per_visit"]
        cart_to_purchase = eng["cart_to_purchase_rate"]
        coupon_rate = eng["coupon_usage_rate"]
        review_rate = eng["review_rate"]
        cs_monthly = eng["cs_contact_monthly"]
        avg_session_minutes = eng["avg_session_minutes"]
        weekend_boost = eng["weekend_activity_boost"]
        target_purchase_frequency = float(
            persona_cfg.get("purchase_frequency_monthly", 0.0)
        )

        # Decay rates
        visit_decay = decay["visit_decay_rate"]
        purchase_cycle_inc = decay["purchase_cycle_increase"]
        session_decay = decay["session_time_decay"]

        # Monthly churn probability (customer may stop all activity)
        base_churn_prob = persona_cfg["churn_probability"]

        # Marketing lifts (only for treatment group)
        coupon_lift = mktg["coupon_conversion_lift"] if is_treatment else 0.0
        push_lift = mktg["push_notification_lift"] if is_treatment else 0.0
        adverse_prob = mktg["adverse_effect_prob"] if is_treatment else 0.0

        avg_order_value = persona_cfg["avg_order_value"]

        # Iterate through each day of the simulation from signup to end
        active_days = (self.end_date - signup_date).days + 1
        if active_days <= 0:
            return events

        # Determine if/when customer explicitly churns (stops all activity)
        # The persona churn_probability represents the overall likelihood of
        # churning during the simulation. We scale it by simulation length
        # relative to a 12-month baseline so that results are stable.
        sim_months = active_days / 30.0
        baseline_months = 12.0
        scale = sim_months / baseline_months

        # Overall churn probability for this customer
        # Use sqrt scaling so shorter simulations still produce adequate
        # churn rates within the 15-25% target range.
        overall_churn_p = base_churn_prob * (scale ** 0.4) * 0.90
        # Treatment can reduce churn but adverse effect may increase it
        if is_treatment:
            overall_churn_p *= (1.0 - coupon_lift * 0.3)
            if self.rng.random() < adverse_prob:
                overall_churn_p *= 1.3

        churn_day = None
        if self.rng.random() < overall_churn_p:
            # Customer will churn; pick a day weighted toward later in timeline
            # Use beta distribution favoring later days (more realistic)
            churn_frac = self.rng.beta(2.0, 1.5)  # Skews toward later
            churn_day = int(churn_frac * active_days)

        for day_offset in range(active_days):
            current_date = signup_date + timedelta(days=day_offset)

            # If customer has churned, stop generating events
            if churn_day is not None and day_offset >= churn_day:
                break

            # Calculate month index for decay
            month_idx = day_offset / 30.0

            # Apply behavior decay
            decayed_visit_prob = daily_visit_prob * max(
                0.01, (1.0 - visit_decay) ** month_idx
            )
            decayed_session_minutes = avg_session_minutes * max(
                0.10, (1.0 - session_decay) ** month_idx
            )

            # Apply weekend boost
            if current_date.weekday() >= 5:  # Saturday=5, Sunday=6
                decayed_visit_prob = min(1.0, decayed_visit_prob * weekend_boost)

            # Apply push notification lift for treatment
            decayed_visit_prob = min(1.0, decayed_visit_prob + push_lift * 0.1)

            # Decide if customer visits today
            if self.rng.random() >= decayed_visit_prob:
                # No visit today; check for cs_contact (can happen without visit)
                cs_daily_prob = cs_monthly / 30.0
                if self.rng.random() < cs_daily_prob:
                    events.append(self._make_event(
                        customer_id, "cs_contact", current_date
                    ))
                continue

            # --- Visit session ---
            # Generate timestamp for the visit
            hour = self.rng.randint(6, 24)
            minute = self.rng.randint(0, 60)
            visit_ts = current_date.replace(hour=hour, minute=minute, second=0)
            session_duration = self._sample_session_duration_seconds(
                decayed_session_minutes
            )
            push_response = self._sample_marketing_response(
                is_treatment,
                response_prob=push_lift,
                adverse_prob=adverse_prob,
            )

            # page_view (every visit generates at least one)
            events.append(self._make_event(
                customer_id, "page_view", current_date, visit_ts,
                session_duration=session_duration,
                marketing_channel=(
                    "push_notification" if push_response is not None else None
                ),
                marketing_response=push_response,
            ))

            # search events
            decayed_search = search_per_visit * max(
                0.1, (1.0 - session_decay) ** month_idx
            )
            n_searches = self.rng.poisson(max(0.1, decayed_search))
            for _ in range(n_searches):
                events.append(self._make_event(
                    customer_id, "search", current_date, visit_ts,
                    session_duration=session_duration,
                ))

            # add_to_cart events
            decayed_cart = cart_add_per_visit * max(
                0.1, (1.0 - session_decay) ** month_idx
            )
            n_cart_adds = self.rng.poisson(max(0.1, decayed_cart))
            for _ in range(n_cart_adds):
                events.append(self._make_event(
                    customer_id, "add_to_cart", current_date, visit_ts,
                    session_duration=session_duration,
                ))

            # remove_from_cart (fraction of cart adds)
            if n_cart_adds > 0:
                n_removals = self.rng.binomial(n_cart_adds, 0.25)
                for _ in range(n_removals):
                    events.append(self._make_event(
                        customer_id, "remove_from_cart", current_date, visit_ts,
                        session_duration=session_duration,
                    ))

            # purchase decision
            if n_cart_adds > 0:
                # Decay purchase probability over time
                conversion_propensity = cart_to_purchase * max(
                    0.12,
                    1.0 - (purchase_cycle_inc / 30.0) * month_idx
                )
                expected_visit_days = max(1.0, 30.0 * decayed_visit_prob)
                frequency_propensity = min(
                    1.0,
                    target_purchase_frequency / expected_visit_days,
                )
                purchase_prob = min(conversion_propensity, frequency_propensity)
                # Add coupon conversion lift for treatment
                purchase_prob = min(1.0, purchase_prob + coupon_lift)

                if self.rng.random() < purchase_prob:
                    # Generate purchase amount with some variance
                    amount = max(
                        1000,
                        self.rng.normal(avg_order_value, avg_order_value * 0.3)
                    )
                    amount = round(amount, -2)  # Round to nearest 100 KRW
                    events.append(self._make_event(
                        customer_id, "purchase", current_date, visit_ts,
                        amount=amount,
                        session_duration=session_duration,
                        marketing_channel="coupon" if is_treatment else None,
                        marketing_response="conversion" if is_treatment else None,
                    ))

                    # Review after purchase
                    if self.rng.random() < review_rate:
                        # Review comes 1-7 days later
                        review_delay = self.rng.randint(1, 8)
                        review_date = current_date + timedelta(days=review_delay)
                        if review_date <= self.end_date:
                            events.append(self._make_event(
                                customer_id, "review", review_date
                            ))

                    # Coupon use (treatment group has higher rate)
                    effective_coupon_rate = coupon_rate
                    if is_treatment:
                        effective_coupon_rate = min(
                            1.0, coupon_rate + coupon_lift
                        )
                    if self.rng.random() < effective_coupon_rate:
                        coupon_response = (
                            "conversion" if is_treatment else None
                        )
                        events.append(self._make_event(
                            customer_id, "coupon_use", current_date, visit_ts,
                            session_duration=session_duration,
                            marketing_channel="coupon" if is_treatment else None,
                            marketing_response=coupon_response,
                        ))

            # cs_contact (can also happen during a visit)
            cs_daily_prob = cs_monthly / 30.0
            if self.rng.random() < cs_daily_prob:
                cs_response = None
                if is_treatment and adverse_prob >= 0.08:
                    cs_response = "adverse"
                events.append(self._make_event(
                    customer_id, "cs_contact", current_date, visit_ts,
                    session_duration=session_duration,
                    marketing_channel="push_notification" if cs_response else None,
                    marketing_response=cs_response,
                ))

        return events

    def _sample_session_duration_seconds(
        self,
        decayed_session_minutes: float,
    ) -> float:
        """Return a positive session duration after persona time decay."""
        return float(round(max(60.0, decayed_session_minutes * 60.0), 2))

    def _sample_marketing_response(
        self,
        is_treatment: bool,
        response_prob: float,
        adverse_prob: float,
    ) -> Optional[str]:
        """Derive treatment response metadata without perturbing event RNG."""
        if not is_treatment:
            return None
        if adverse_prob >= 0.08:
            return "adverse"
        if response_prob >= 0.10:
            return "conversion"
        if response_prob > 0:
            return "no_response"
        return None

    def _make_event(
        self,
        customer_id: str,
        event_type: str,
        event_date: pd.Timestamp,
        timestamp: Optional[pd.Timestamp] = None,
        amount: Optional[float] = None,
        session_duration: Optional[float] = None,
        marketing_channel: Optional[str] = None,
        marketing_response: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an event record dictionary.

        Args:
            customer_id: Customer identifier.
            event_type: Type of event.
            event_date: Date of the event.
            timestamp: Optional precise timestamp. If None, generated randomly.
            amount: Optional monetary amount (for purchase events).
            session_duration: Optional session duration in seconds.
            marketing_channel: Optional intervention channel.
            marketing_response: Optional treatment response category.

        Returns:
            Event dictionary.
        """
        if timestamp is None:
            hour = self.rng.randint(6, 24)
            minute = self.rng.randint(0, 60)
            second = self.rng.randint(0, 60)
            timestamp = event_date.replace(
                hour=hour, minute=minute, second=second
            )

        return {
            "customer_id": customer_id,
            "event_type": event_type,
            "event_date": event_date.strftime("%Y-%m-%d"),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amount,
            "session_duration": session_duration,
            "marketing_channel": marketing_channel,
            "marketing_response": marketing_response,
        }

    # ------------------------------------------------------------------
    # Churn Labeling
    # ------------------------------------------------------------------

    def _label_churn(
        self,
        customers_df: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Label customers as churned or active based on configurable rules.

        Churn definition (configurable):
        - no_purchase_days: Days without purchase (default 30)
        - no_login_days: Days without login/visit (default 60)
        - operator: OR (either condition) or AND (both conditions)

        Uses vectorized pandas groupby operations for performance.

        Args:
            customers_df: Customer profiles DataFrame.
            events_df: Events DataFrame.

        Returns:
            Updated customers DataFrame with 'churn_label' column.
        """
        customers_df = customers_df.copy()
        ref_date = self.end_date

        if len(events_df) == 0:
            customers_df["churn_label"] = 1
            return customers_df

        events_tmp = events_df.copy()
        events_tmp["event_date_dt"] = pd.to_datetime(events_tmp["event_date"])

        # Last purchase date per customer (vectorized)
        purchases = events_tmp[events_tmp["event_type"] == "purchase"]
        last_purchase = (
            purchases.groupby("customer_id")["event_date_dt"]
            .max()
            .rename("last_purchase_date")
        )

        # Last visit date per customer (vectorized)
        visits = events_tmp[
            events_tmp["event_type"].isin(["page_view", "search"])
        ]
        last_visit = (
            visits.groupby("customer_id")["event_date_dt"]
            .max()
            .rename("last_visit_date")
        )

        # Merge with customers
        customers_df = customers_df.merge(
            last_purchase, left_on="customer_id", right_index=True, how="left"
        )
        customers_df = customers_df.merge(
            last_visit, left_on="customer_id", right_index=True, how="left"
        )

        # Compute days since last activity
        days_since_purchase = (
            ref_date - customers_df["last_purchase_date"]
        ).dt.days.fillna(self.simulation_days)
        days_since_visit = (
            ref_date - customers_df["last_visit_date"]
        ).dt.days.fillna(self.simulation_days)

        # Apply churn definition
        no_purchase_churn = days_since_purchase >= self.no_purchase_days
        no_login_churn = days_since_visit >= self.no_login_days

        if self.churn_operator == "OR":
            is_churned = no_purchase_churn | no_login_churn
        else:  # AND
            is_churned = no_purchase_churn & no_login_churn

        customers_df["churn_label"] = is_churned.astype(int)
        customers_df = customers_df.drop(
            columns=["last_purchase_date", "last_visit_date"]
        )
        return customers_df
