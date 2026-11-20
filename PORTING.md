# Provenance of ported code

Source: fusionlab-learn, branch `develop`, commit `92e4aa6`
(`E:\_REPOSITORIES\fusionlab-learn`, clean working tree), copied on
2026-11-20 with `tools/port_from_fusionlab.py`.

The script copies the import closure of the Keras XTFT, SuperXTFT
and TFT, plus the support layer (decorators, compat, backends,
api, core, utils, metrics). It rewrites every import, relative or
absolute, to an absolute `xtft.*` path:

| fusionlab | xtft |
|---|---|
| `fusionlab.nn.*` | `xtft.nn.tf.*` |
| `fusionlab.backends.*` (array backends) | `xtft.backends.array.*` |
| `fusionlab._fusionlog` | `xtft._log` |
| `fusionlab.*` | `xtft.*` |

Deliberately excluded: `nn.pinn` (GeoPrior/PINN), `nn.forecast_tuner`,
`nn.hybrid._halnet`, `nn.utils`, `utils.nat_utils`,
`utils.subsidence_utils`, `utils.scale_metrics`. The package
`__init__` files were rewritten slim (fusionlab's pulled in every
subpackage).

## Changes made after copying

* `utils/forecast_utils.py`: GeoPrior-only helpers
  `_resolve_stage1_entry` and `convert_target_units_df` replaced by
  stubs that raise `NotImplementedError`.
* `nn/tf/hybrid/_base_extreme.py`: `set_default_params` imported
  from `xtft.nn.tf._utils`.
* Keras 3 bugs fixed (they also exist in fusionlab):
  * `add_weight("name", shape=...)` passed the name as `shape`
    (`components/gating_norm.py`, `components/layer_utils.py`,
    `keras_metrics.py`); now `add_weight(name=..., ...)`.
  * `TemporalAttentionLayer.build` rejected a plain shape tuple
    (`components/attention.py`); TFT failed on its first call.
* `nn/tf/_tensor_validation.py`: `rank is -1` changed to
  `rank == -1`.
* Dotted `fusionlab.` paths in docstrings renamed to `xtft.`.

## Known leftovers (to clean when each module is rewritten)

* Identifiers `fusionlog`, `get_fusionlab_logger` and prose that
  mentions FusionLab.
* Legacy docstring examples that are not runnable (for example
  `from xtft.nn.tf.transformers import XTFT`, local paths).
* Ported code does not yet meet the 71-character and numpydoc
  standards; Ruff excludes it (see `pyproject.toml`).

## Module map

| fusionlab module | xtft file |
|---|---|
| `fusionlab._configs` | `xtft\_configs.py` |
| `fusionlab._deps` | `xtft\_deps.py` |
| `fusionlab._dummies` | `xtft\_dummies.py` |
| `fusionlab._fusionlog` | `xtft\_log.py` |
| `fusionlab._optdeps` | `xtft\_optdeps.py` |
| `fusionlab._util` | `xtft\_util.py` |
| `fusionlab.api.bunch` | `xtft\api\bunch.py` |
| `fusionlab.api.docs` | `xtft\api\docs.py` |
| `fusionlab.api.property` | `xtft\api\property.py` |
| `fusionlab.api.summary` | `xtft\api\summary.py` |
| `fusionlab.api.types` | `xtft\api\types.py` |
| `fusionlab.api.util` | `xtft\api\util.py` |
| `fusionlab.backends.base` | `xtft\backends\array\base.py` |
| `fusionlab.backends.cupy` | `xtft\backends\array\cupy.py` |
| `fusionlab.backends.dask` | `xtft\backends\array\dask.py` |
| `fusionlab.backends.devices` | `xtft\backends\array\devices.py` |
| `fusionlab.backends.nn` | `xtft\backends\array\nn.py` |
| `fusionlab.backends.numpy` | `xtft\backends\array\numpy.py` |
| `fusionlab.backends.scipy` | `xtft\backends\array\scipy.py` |
| `fusionlab.backends.selector` | `xtft\backends\array\selector.py` |
| `fusionlab.compat` | `xtft\compat\__init__.py` |
| `fusionlab.compat._config` | `xtft\compat\_config.py` |
| `fusionlab.compat.keras` | `xtft\compat\keras.py` |
| `fusionlab.compat.keras_fit` | `xtft\compat\keras_fit.py` |
| `fusionlab.compat.kt` | `xtft\compat\kt.py` |
| `fusionlab.compat.numpy` | `xtft\compat\numpy.py` |
| `fusionlab.compat.pandas` | `xtft\compat\pandas.py` |
| `fusionlab.compat.scipy` | `xtft\compat\scipy.py` |
| `fusionlab.compat.sklearn` | `xtft\compat\sklearn.py` |
| `fusionlab.compat.tf` | `xtft\compat\tf.py` |
| `fusionlab.compat.types` | `xtft\compat\types.py` |
| `fusionlab.core.array_manager` | `xtft\core\array_manager.py` |
| `fusionlab.core.checks` | `xtft\core\checks.py` |
| `fusionlab.core.diagnose_q` | `xtft\core\diagnose_q.py` |
| `fusionlab.core.handlers` | `xtft\core\handlers.py` |
| `fusionlab.core.io` | `xtft\core\io.py` |
| `fusionlab.core.utils` | `xtft\core\utils.py` |
| `fusionlab.decorators` | `xtft\decorators.py` |
| `fusionlab.exceptions` | `xtft\exceptions.py` |
| `fusionlab.metrics` | `xtft\metrics\__init__.py` |
| `fusionlab.metrics._cas` | `xtft\metrics\_cas.py` |
| `fusionlab.metrics._metrics` | `xtft\metrics\_metrics.py` |
| `fusionlab.metrics._registry` | `xtft\metrics\_registry.py` |
| `fusionlab.nn` | `xtft\nn\tf\__init__.py` |
| `fusionlab.nn._shapes` | `xtft\nn\tf\_shapes.py` |
| `fusionlab.nn._tensor_validation` | `xtft\nn\tf\_tensor_validation.py` |
| `fusionlab.nn._utils` | `xtft\nn\tf\_utils.py` |
| `fusionlab.nn.anomaly_detection` | `xtft\nn\tf\anomaly_detection.py` |
| `fusionlab.nn.calibration` | `xtft\nn\tf\calibration.py` |
| `fusionlab.nn.callbacks` | `xtft\nn\tf\callbacks.py` |
| `fusionlab.nn.components` | `xtft\nn\tf\components\__init__.py` |
| `fusionlab.nn.components._attention_utils` | `xtft\nn\tf\components\_attention_utils.py` |
| `fusionlab.nn.components._config` | `xtft\nn\tf\components\_config.py` |
| `fusionlab.nn.components._loss_utils` | `xtft\nn\tf\components\_loss_utils.py` |
| `fusionlab.nn.components._temporal_utils` | `xtft\nn\tf\components\_temporal_utils.py` |
| `fusionlab.nn.components.attention` | `xtft\nn\tf\components\attention.py` |
| `fusionlab.nn.components.encoder_decoder` | `xtft\nn\tf\components\encoder_decoder.py` |
| `fusionlab.nn.components.gating_norm` | `xtft\nn\tf\components\gating_norm.py` |
| `fusionlab.nn.components.heads` | `xtft\nn\tf\components\heads.py` |
| `fusionlab.nn.components.layer_utils` | `xtft\nn\tf\components\layer_utils.py` |
| `fusionlab.nn.components.losses` | `xtft\nn\tf\components\losses.py` |
| `fusionlab.nn.components.masks` | `xtft\nn\tf\components\masks.py` |
| `fusionlab.nn.components.misc` | `xtft\nn\tf\components\misc.py` |
| `fusionlab.nn.components.temporal` | `xtft\nn\tf\components\temporal.py` |
| `fusionlab.nn.components.utils` | `xtft\nn\tf\components\utils.py` |
| `fusionlab.nn.custom_metrics` | `xtft\nn\tf\custom_metrics.py` |
| `fusionlab.nn.hybrid._base_extreme` | `xtft\nn\tf\hybrid\_base_extreme.py` |
| `fusionlab.nn.hybrid._super_xtft` | `xtft\nn\tf\hybrid\_super_xtft.py` |
| `fusionlab.nn.hybrid._xtft` | `xtft\nn\tf\hybrid\_xtft.py` |
| `fusionlab.nn.hybrid.utils` | `xtft\nn\tf\hybrid\utils.py` |
| `fusionlab.nn.keras_metrics` | `xtft\nn\tf\keras_metrics.py` |
| `fusionlab.nn.keras_validator` | `xtft\nn\tf\keras_validator.py` |
| `fusionlab.nn.losses` | `xtft\nn\tf\losses.py` |
| `fusionlab.nn.transformers._adj_tft` | `xtft\nn\tf\transformers\_adj_tft.py` |
| `fusionlab.nn.transformers._tft` | `xtft\nn\tf\transformers\_tft.py` |
| `fusionlab.utils._array_api` | `xtft\utils\_array_api.py` |
| `fusionlab.utils._dependency` | `xtft\utils\_dependency.py` |
| `fusionlab.utils.base_utils` | `xtft\utils\base_utils.py` |
| `fusionlab.utils.calibrate` | `xtft\utils\calibrate.py` |
| `fusionlab.utils.data_utils` | `xtft\utils\data_utils.py` |
| `fusionlab.utils.deps_utils` | `xtft\utils\deps_utils.py` |
| `fusionlab.utils.forecast_utils` | `xtft\utils\forecast_utils.py` |
| `fusionlab.utils.generic_utils` | `xtft\utils\generic_utils.py` |
| `fusionlab.utils.io_utils` | `xtft\utils\io_utils.py` |
| `fusionlab.utils.scale_metrics` | `xtft\utils\scale_metrics.py` |
| `fusionlab.utils.sequence_utils` | `xtft\utils\sequence_utils.py` |
| `fusionlab.utils.shapes` | `xtft\utils\shapes.py` |
| `fusionlab.utils.sys_utils` | `xtft\utils\sys_utils.py` |
| `fusionlab.utils.ts_utils` | `xtft\utils\ts_utils.py` |
| `fusionlab.utils.validator` | `xtft\utils\validator.py` |
| `fusionlab.utils.version` | `xtft\utils\version.py` |
