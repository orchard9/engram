def alias_window:
  . as $win
  | $win
  + {
      pool_available: $win.activation_pool_available_records,
      pool_in_flight: $win.activation_pool_in_flight_records,
      pool_high_water: $win.activation_pool_high_water_mark,
      pool_total_created: $win.activation_pool_total_created,
      pool_total_reused: $win.activation_pool_total_reused,
      pool_miss_count: $win.activation_pool_miss_count,
      pool_release_failures: $win.activation_pool_release_failures,
      pool_hit_rate: $win.activation_pool_hit_rate,
      pool_utilization: $win.activation_pool_utilization
    }
  | with_entries(select(.value != null));

.snapshot |= (
  with_entries(
    if .key == "schema_version" then . else .value |= alias_window end
  )
)
