# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2024-01-09

### Added
- **Cross-DC Training Support**: New capability to model performance impact of cross-datacenter training
  - Added `cross_dc` configuration section to model multi-DC deployments
  - Supports ring topology with configurable number of DCs, interconnect bandwidth, and latency
  - Calculates and reports DP communication degradation due to cross-DC links
  - Cross-DC overhead is reflected in final MFU calculations
  - Example configuration: `examples/llama3_70b_cross_dc.yaml`

### Changed
- **Breaking Change**: Replaced `bucket_size_mb` with `n_param_buckets` in parallelism configuration
  - Users now specify the number of gradient buckets directly instead of bucket size in MB
  - More intuitive interface for controlling communication rounds
  - Bucket size is automatically calculated as total params / n_param_buckets

### Technical Details
- Added `CrossDCConfig` dataclass in `dlcalc.utils.configurations`
- New cross-DC aware communication functions in `dlcalc.utils.comms`:
  - `get_cross_dc_dp_all_gather_comm_time_s()`
  - `get_cross_dc_dp_reduce_scatter_comm_time_s()`
- Models heterogeneous ring topology with mixed cross-DC and inter-node latencies
- Correctly accounts for bandwidth throttling due to slower cross-DC links

### Migration Guide
Replace `bucket_size_mb` with `n_param_buckets` in your configuration files:

**Before:**
```yaml
parallelism:
  bucket_size_mb: 250
```

**After:**
```yaml
parallelism:
  n_param_buckets: 5
```

To enable cross-DC modeling, add the optional `cross_dc` section:
```yaml
cross_dc:
  n_dcs: 2
  interconnect_bandwidth_gbps: 100000
  interconnect_latency_s: 0.005
```

## [0.1.11] - Previous Release
- Previous release notes...
