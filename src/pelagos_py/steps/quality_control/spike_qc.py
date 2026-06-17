# This file is part of pelagos_py.
#
# Copyright 2025-2026 National Oceanography Centre and The Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QC test for flagging using spike/despike detection methods."""

#### Mandatory imports ####
import numpy as np
from pelagos_py.steps.base_qc import BaseQC, register_qc, flag_cols

#### Custom imports ####
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib
from tqdm import tqdm


@register_qc
class spike_qc(BaseQC):
    """
    Target Variable: Any
    Flag Number: 4 (bad)
    Variables Flagged: Any
    Checks for spiking in the data using rolling median values compared against the
    meadian average deviation (MAD).

    EXAMPLE
    -------
    ::

        - name: "Apply QC"
          parameters:
            qc_settings: {
                "spike test": {
                  "variables": {"PRES": 2, "LATITUDE": 1},
                  "also_flag": {"PRES": ["CNDC", "TEMP"], "LATITUDE": ["LONGITUDE"]},
                  "plot": ["PRES", "LATITUDE"]
                  "window_size": 10,
                }
            }
          diagnostics: true
    """

    qc_name = "spike qc"

    # Specify if test target variable is user-defined (if True, __init__ has to be redefined)
    dynamic = True

    def __init__(self, data, **kwargs):
        #   Called when apply QC checks the required variables
        # Check the necessary kwargs are available
        required_kwargs = {
            "variables",
            "method",
            "by_profile",
            "also_flag",
            "plot",
        }  #   TODO: Make method, also_flag, and plot optional
        if not required_kwargs.issubset(set(kwargs.keys())):
            raise KeyError(
                f"{required_kwargs - set(kwargs.keys())} are missing from {self.qc_name} settings"
            )

        # Specify the tests paramters from kwargs (config)
        self.expected_parameters = {
            k: v for k, v in kwargs.items() if k in required_kwargs
        }
        self.required_variables = list(
            set(self.expected_parameters["variables"].keys())
        )
        if ("by_profile" in kwargs) and (kwargs["by_profile"] == True):
            self.required_variables += ["PROFILE_NUMBER"]

        self.qc_outputs = list(
            set(f"{var}_QC" for var in self.required_variables)
            | set(
                f"{var}_QC"
                for var in sum(self.expected_parameters["also_flag"].values(), [])
            )
        )

        if data is not None:
            self.data = data.copy(deep=True)

        # set attributes
        for k, v in self.expected_parameters.items():
            setattr(self, k, v)
        self.window_size = kwargs.get("window_size") or 50

        self.flags = None

    def return_qc(self):
        #   New function
        self.data = self.data[self.required_variables]
        profile_idxs = self.slice_profiles()

        for var, cond in self.variables.items():
            #   Runs the specified variable var with sensitivity tolerance
            var_data = self.data[var]
            new_flags = np.full(len(var_data), 0)  #   Init to 0, not assessed
            
            for prof in profile_idxs:
                data_pass_on = var_data[prof[0] : prof[1]]
                
                if self.method == "rolling median":
                    spike_flags = self.rolling_median(
                        data=data_pass_on, window=self.window_size, sensitivity=cond
                    )
                elif self.method == "qartod":
                    spike_flags = self.qartod_despike_diff(
                        data=data_pass_on, thresh=cond
                    )
                elif self.method == "hampel":
                    spike_flags = self.hampel_despike(
                        data=data_pass_on, window=self.window_size, nsigma=cond
                    )

                new_flags[prof[0] : prof[1]] = spike_flags

            nan_mask = np.isnan(var_data)
            # if self.by_profile:   #   Uncomment to include unknown profile numbers as NaNs
            #     nan_mask += np.isnan(self.data["PROFILE_NUMBER"])
            profile_flags = new_flags.copy()
            profile_flags[nan_mask] = 9
            if any(profile_flags == 0):
                #   Values are not nan, but there are gaps in the profile detection
                self.log_warn(
                    f"Despike method '{self.method}' has untested data (flag 0={list(profile_flags).count(0)}) following the test for {var}.\n"
                    f"Consider running on full series or check profile numbers."
                )

            self.data[f"{var}_QC"] = (["N_MEASUREMENTS"], profile_flags)

            if extra_vars := self.also_flag.get(var):
                for extra_var in extra_vars:
                    self.data[f"{extra_var}_QC"] = self.data[f"{var}_QC"]

        self.flags = self.data[
            [var_qc for var_qc in self.data.data_vars if "_QC" in var_qc]
        ]

        return self.flags

    def slice_profiles(self):
        profile_idxs = list()
        if self.by_profile:
            profiles = self.data["PROFILE_NUMBER"].values

            profile_idxs = [
                (idxs[0].item(), idxs[-1].item() + 1)
                for p in np.unique(profiles[~np.isnan(profiles)])
                if (idxs := np.flatnonzero(profiles == p)).size
            ]
        else:
            profile_idxs.append(
                (0, len(self.data["N_MEASUREMENTS"]) - 1)
            )  #   If not doing profile-by-profile, select whole thing
        
        return profile_idxs

    def rolling_median(
        self, data: np.ndarray, window: int = 10, sensitivity: int = 2
    ) -> np.ndarray:  #   Default spike qc for pelagos
        rolling_median = (
            data.to_pandas().rolling(window=window, center=True).median().to_numpy()
        )
        residules = data - rolling_median

        # Define the residule threshold
        threshold = np.nanstd(residules) * sensitivity

        # Apply the threshold to residules to get the flags
        spike_flags = np.where((np.abs(residules) > threshold), 4, 1)
        return spike_flags

    def qartod_despike_diff(self, data: np.ndarray, thresh=0.02) -> np.ndarray:
        from ioos_qc import spike_test as qartod_spike

        flags_qartod = qartod_spike(
            inp=data, fail_threshold=thresh, method="differential"
        )
        flags = np.zeros(shape=flags_qartod.shape)
        flags[np.where(flags_qartod == 4)] = 1
        return flags

    def hampel_despike(
        self, data: np.ndarray, window: int = 3, nsigma: float = 3.0
    ) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        n = len(data)
        half = window // 2
        mask = np.zeros(n, dtype=bool)
        scale = 1.4826

        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            window_data = data[lo:hi]
            med = np.median(window_data)
            mad = np.median(np.abs(window_data - med))
            if np.abs(data[i] - med) > nsigma * scale * mad:
                mask[i] = True

        return mask

    def plot_diagnostics(self):
        matplotlib.use("tkagg")

        # If not plots were specified
        if len(self.plot) == 0:
            self.log_warn(
                f"WARNING: In '{self.qc_name}', diagnostics were called but no variables were specified for plotting."
            )
            return

        # Plot the QC output
        fig, axs = plt.subplots(
            nrows=len(self.plot), figsize=(8, 6), sharex=True, dpi=200
        )
        if len(self.plot) == 1:
            axs = [axs]

        for ax, var in zip(axs, self.plot):
            # Check that the user specified var exists in the test set
            if f"{var}_QC" not in self.qc_outputs:
                print(
                    f"WARNING: Cannot plot {var}_QC as it was not included in this test."
                )
                continue

            for i in range(10):
                # Plot by flag number
                plot_data = self.data[[var, "N_MEASUREMENTS"]].where(
                    self.data[f"{var}_QC"] == i, drop=True
                )

                if len(plot_data[var]) == 0:
                    continue

                # Plot the data
                ax.plot(
                    plot_data["N_MEASUREMENTS"],
                    plot_data[var],
                    c=flag_cols[i],
                    ls="",
                    marker="o",
                    label=f"{i}",
                )

            ax.set(
                xlabel="Index",
                ylabel=var,
                title=f"{var} Spike Test",
            )

            ax.legend(title="Flags", loc="upper right")

        fig.tight_layout()
        plt.show(block=True)

### Legacy code ###
# def return_qc_1(self):
#     # Subset the data
#     self.data = self.data[self.required_variables]
#     # Generate the variable-specific flags
#     for var, sensitivity in self.variables.items():
#         spike_qc = np.full(len(self.data[var]), 0)

#         # Apply the checks across individual profiles
#         profile_numbers = np.unique(
#             self.data["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS")
#         )
#         for profile_number in tqdm(
#             profile_numbers,
#             colour="green",
#             desc=f"\033[97mProgress [{var}]\033[0m",
#             unit="prof",
#         ):
#             # Subset the data
#             profile = self.data.where(
#                 self.data["PROFILE_NUMBER"] == profile_number, drop=True
#             )

#             # remove nans
#             var_data = profile[var].dropna(dim="N_MEASUREMENTS")
#             if len(var_data) < self.window_size:
#                 continue

#             # Calculate the residules from the running median of the data
#             rolling_median = (
#                 var_data.to_pandas()
#                 .rolling(window=self.window_size, center=True)
#                 .median()
#                 .to_numpy()
#             )
#             residules = var_data - rolling_median

#             # Define the residule threshold
#             threshold = np.nanstd(residules) * sensitivity

#             # Apply the threshold to residules to get the flags
#             spike_flags = np.where((np.abs(residules) > threshold), 4, 1)

#             # Reinclude the nans as missing (9) flags
#             nan_mask = np.isnan(profile[var])
#             profile_flags = np.where(nan_mask, 9, 1)
#             profile_flags[np.where(~nan_mask)] = spike_flags

#             # Stitch the QC results back into the QC container
#             profile_indices = np.where(
#                 self.data["PROFILE_NUMBER"] == profile_number
#             )
#             spike_qc[profile_indices] = profile_flags

#         # Add the flags to the data
#         self.data[f"{var}_QC"] = (["N_MEASUREMENTS"], spike_qc)

#         # Broadcast the QC found for var into variables specified by "also_flag"
#         if extra_vars := self.also_flag.get(var):
#             for extra_var in extra_vars:
#                 self.data[f"{extra_var}_QC"] = self.data[f"{var}_QC"]

#     # Select just the flags
#     self.flags = self.data[
#         [var_qc for var_qc in self.data.data_vars if "_QC" in var_qc]
#     ]

#     return self.flags