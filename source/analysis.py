"""
@context
    * Performs various analyses on preprocessed data
    * Data must be preprocessed before analysing
"""


import matplotlib.pyplot       as plt
import numpy                   as np
import os
import pandas                  as pd
import statsmodels.api         as sm
import statsmodels.formula.api as smf

from collections             import defaultdict
from scipy.stats             import chi2, gaussian_kde, norm
from sklearn.metrics         import root_mean_squared_error
from sklearn.model_selection import KFold
from zipfile                 import ZipFile


DIR_DATASETS = "datasets"
DIR_PREPROCESSED = "datasets/curated/preprocessed.csv"

DIR_SAVE_BOXPLOT       = "plots/boxplot_{on}.png"
DIR_SAVE_RELATIONSHIPS = "plots/relationships.png"
DIR_SAVE_DIAGNOSTICS   = "plots/diagnostics.png"

# * Effects in the initial model formula
# * Formula starts with two-way interactions between all main effects
# * Many of these effects will be automatically dropped by both backwards
#   stepwise selection and their p-value significance
# * Main effects cannot be dropped if being used in an interaction effect
DEPENDENT_VAR = "mean_rent"
MAIN_EFFECTS = [
    "employed",
    "median_household_income",
    "transport_methods_1",
    "transport_only_car",
    "worked_home"
]
INTERACTION_EFFECTS = [
    "employed:median_household_income",
    "employed:transport_methods_1",
    "employed:transport_only_car",
    "employed:worked_home",

    "median_household_income:transport_methods_1",
    "median_household_income:transport_only_car",
    "median_household_income:worked_home",

    "transport_methods_1:transport_only_car",
    "transport_methods_1:worked_home",

    "transport_only_car:worked_home",
]

TITLE_RELATIONSHIPS = "Pairwise Relationships"
TITLE_DIAGNOSTICS = "Diagnostic Plots"

TITLE_RESIDUALS_VS_FITTED = "Residuals vs. Fitted"
TITLE_NORMAL_QQ = "Normal Q-Q"
TITLE_SCALE_LOCATION = "Scale-Location"
TITLE_RESIDUALS_VS_LEVERAGE = "Residuals vs. Leverage"

LABEL_X_RESIDUALS_VS_FITTED = "Fitted Values"
LABEL_X_NORMAL_QQ = "Theoretical Quantiles"
LABEL_X_SCALE_LOCATION = "Fitted Values"
LABEL_X_RESIDUALS_VS_LEVERAGE = "Leverage"

LABEL_Y_RESIDUALS_VS_FITTED = "Residuals"
LABEL_Y_NORMAL_QQ = "Ordered Standardised Residuals"
LABEL_Y_SCALE_LOCATION = r"$\sqrt{\left|\text{Standardised Residuals}\right|}$"
LABEL_Y_RESIDUALS_VS_LEVERAGE = "Standardised Residuals"

# * Plot sizes
DIAGNOSTIC_SIZE   = (20, 12)
RELATIONSHIP_SIZE = (20, 15)
BOXPLOT_SIZE      = (10,  5)

# * Plot text sizes
TITLE_SIZE   = 30
SUBTILE_SIZE = 20
LABEL_SIZE   = 13

# * Image quality of plots
PLOT_DPI = 150

# * Any effects with a p-value < `P_VALUE_NON_SIGNIFICANT` are dropped
P_VALUE_NON_SIGNIFICANT = 0.01

# * Reject null hypothesis if p-value < `P_VALUE_ADEQUATE`
P_VALUE_ADEQUATE = 0.01

# * Whether to perform leave-one-out cross-validation
# * If `False` then perform k-fold cross-validation with `K_FOLDS` folds
IS_LEAVE_ONE_OUT = True
K_FOLDS = 8

# * Smoothing parameter of LOWESS lines
LOWESS_SMOOTHING = 0.75


def main():
    """
    @context
        * Performs various analyses on preprocessed data
    """

    # * Unzip the dataset folder if it has not already been unzipped
    if not os.path.exists(DIR_DATASETS):
        print("Extracting zipped datasets")
        with ZipFile(f"{DIR_DATASETS}.zip", "r") as file:
            file.extractall(".")
        print("Extracted zipped datasets")

    if not os.path.exists(DIR_PREPROCESSED):
        print("Need to preprocess rental property data before analysing")
        return

    df = pd.read_csv(DIR_PREPROCESSED)

    # * Display correlation of all `MAIN_EFFECTS` to the `DEPENDENT_VAR`
    print(f"Pearson's correlation coefficient with {DEPENDENT_VAR}:")
    for var in MAIN_EFFECTS:
        pearson_corr(df, var)

    # * Formula starts with all main effects and with two-way interactions
    #   between them
    # * Effects are dropped to find a reduced formula
    effects = MAIN_EFFECTS + INTERACTION_EFFECTS
    effects = drop_aic(df, effects)
    effects = drop_p_value(df, effects)
    formula = get_formula(effects)

    # * Evaluate the reduced `formula` and fit the model with all instances
    model_testing(df, formula)
    model = smf.glm(formula, data=df).fit()
    model_adequacy(model)

    # * Plot of data and `model` diagnostics
    plot_boxplot(df, DEPENDENT_VAR)
    plot_relationships(df)
    plot_diagnostics(model)

    print(f"\n{model.summary()}")


def pearson_corr(df, effect):
    """
    @context
        * Displays the Pearson's correlation of `effect` with `DEPENDENT_VAR`

    @parameters
        * df : Dataframe
            * Dataframe containing `effect` and `DEPENDENT_VAR`
        * effect : str
            * Column in `df` to calculate Pearson's correlation with
    """
    corr = df[DEPENDENT_VAR].corr(df[effect])
    print(f"    {effect:<23}: {corr:6.3f}")


def get_formula(effects):
    """
    @context
        * Creates a GLM formula between `DEPENDENT_VAR` and `effects`

    @parameters
        * effects : list[str]
            * Effects to use in the model

    @return : str
        * GLM formula
    """
    return f"{DEPENDENT_VAR} ~ {' + '.join(effects)}"


def filter_effects(filter_effect, effects):
    """
    @context
        * Removes `filter_effect` from `effects`

    @parameters
        * filter_effect : str
            * Effect to remove from `effects`
        * effects : list[str]
            * Effect to remove `filter_effect` from

    @return : list[str]
        * Effects without `filter_effect`
    """
    return [effect for effect in effects if effect != filter_effect]


def is_effect_droppable(drop_effect, effects):
    """
    @context
        * Determines whether `drop_effect` is droppable from `effects`
        * Main effects cannot be dropped if being used in a interaction effect

    @parameters
        * drop_effect : str
            * Effect to determine whether it is droppable
        * effects : list[str]
            * Effects to determine if `drop_effect` is droppable within

    @return : bool
        * Indicates whether `drop_effect` is droppable from `effects`
    """
    return len([effect for effect in effects if drop_effect in effect]) == 1


def drop_aic(df, effects):
    """
    @context
        * Finds a reduced GLM formula of all `effects` with backwards stepwise
          selection
        * Each step, the single effect whose removal produces a model with the
          smallest AIC is dropped
            * Repeat until no more droppable effects can be dropped to improve
              the model

    @parameters
        * df : Dataframe
            * Dataframe of all main effects in `effects`
        * effects : list[str]
            * Initial effects to reduce from
            * Can contain main and interaction effects

    @return : list[str]
        * Effects that cannot be dropped by backward stepwise selection
    """

    print("\nBackwards stepwise selection:")

    # * Fit the full model (all `effects`) to reduce from
    formula = get_formula(effects)
    model = smf.glm(formula, data=df).fit()
    smallest_aic = model.aic
    effect_with_smallest_aic = None

    # * Drop effects until it does not improve the model's AIC
    while True:
        for effect in effects:

            # * Can only drop a main effect if not being used in an interaction
            if not is_effect_droppable(effect, effects):
                continue

            # * Fit the model with `effect` dropped
            formula = get_formula(filter_effects(effect, effects))
            model = smf.glm(formula, data=df).fit()
            aic = model.aic

            # * Remember `effect` if dropping it improves the model's AIC
            if aic < smallest_aic:
                smallest_aic = aic
                effect_with_smallest_aic = effect

        # * Formula found if the model cannot be improved by dropping an effect
        if not effect_with_smallest_aic:
            break

        effects = filter_effects(effect_with_smallest_aic, effects)
        print(f"    {effect_with_smallest_aic:<42} dropped")
        effect_with_smallest_aic = None

    return effects


def drop_p_value(df, effects):
    """
    @context
        * Finds a reduced GLM formula of all `effects` by the effect p-values
        * Each step, the single most non-significant effect is dropped
            * Repeat until all droppable effects are significant

    @parameters
        * df : Dataframe
             * Dataframe of all main effects in `effects`
        * effects : list[str]
            * Initial effects to reduce from
            * Can contain main and interaction effects

    @return : list[str]
        * Effects that are significant
    """

    print("\nNon-significance dropping:")

    # * Drop effects until all significant
    while True:

        # * Fit the model with all current `effects`
        formula = get_formula(effects)
        model = smf.glm(formula, data=df).fit()
        largest_p_value = 0
        effect_with_largest_p_value = None

        # * Find the most non-significant effect
        for effect in effects:

            # * Can only drop a main effect if not being used in an interaction
            if not is_effect_droppable(effect, effects):
                continue

            p_value = model.pvalues.loc[effect]

            # * Remember the `effect` with the largest p-value
            if p_value > largest_p_value:
                largest_p_value = p_value
                effect_with_largest_p_value = effect

        # * Formula found if all effects are significant
        if (not effect_with_largest_p_value
            or largest_p_value < P_VALUE_NON_SIGNIFICANT):
            break

        effects = filter_effects(effect_with_largest_p_value, effects)
        print(f"    {effect_with_largest_p_value:<38} dropped")

    return effects


def model_testing(df, formula):
    """
    @context
        * Evaluates the model defined by `formula`
        * Uses Root Mean Squared Error (RMSE) through cross-validation

    @parameters
        * df : Dataframe
            * Dataset of all main effects used in `formula`
        * formula : str
            * Model formula to evaluate
    """

    print("\nTesting Model")

    # * Create splits for cross-validation
    if IS_LEAVE_ONE_OUT:
        # * Due to the small number of instances this should not take too long
        n_folds = len(df.index)
        print(f"Performing leave-one-out cross-validation ({n_folds} folds)")
    else:
        n_folds = K_FOLDS
        print(f"Performing cross-validation with {n_folds} folds")
    kf = KFold(n_splits=n_folds)

    # * For each fold fit and collect statistics on the model
    statistics = defaultdict(list)
    for train_index, test_index in kf.split(df):

        df_train = df.iloc[train_index]
        df_test  = df.iloc[test_index]

        # * Fit model and predict current test values
        model = smf.glm(formula, data=df_train).fit()
        predicted = model.predict(df_test)

        rmse = root_mean_squared_error(df_test[DEPENDENT_VAR], predicted)

        # * Statistics about current `model` to keep
        statistics["RMSEs"].append(rmse)
        statistics["Predictions"] += predicted.tolist()

    # * For each statistic display a short summary
    for statistic_name, statistic_values in statistics.items():
        print(
            f"\n{statistic_name}:\n"
            f"    min:    {min(statistic_values):7.3f}\n"
            f"    median: {np.median(statistic_values):7.3f}\n"
            f"    mean:   {np.mean(statistic_values):7.3f}\n"
            f"    max:    {max(statistic_values):7.3f}"
        )


def model_adequacy(model):
    """
    @context
        * Test adequacy of `model`
        * Null hypothesis (`H_0: betas = 0`)
            * Tests if the null model is adequate
            * If rejected it suggests `model` significantly adequate

    @parameters
        * model : Model
            * Model to test adequacy of
    """

    # * `+1` as `statsmodels` does not count the intercept in the df
    # * `-1` as the df of the null models is subtracted
    df = (model.df_model + 1) - 1

    likelihood_ratio = 2 * (model.llf - model.llnull)
    p_value = chi2.sf(likelihood_ratio, df)

    if (p_value < P_VALUE_ADEQUATE):
        print("\nThe model is significantly adequate")
    else:
        print("\nCannot say the model is significantly adequate")


def plot_boxplot(df, on):
    """
    @context
        * Creates and saves a boxplot for the feature `on` in `df`

    @parameters
        * df : Dataframe
            * Dataframe containing `on`
        * on : str
            * Feature to create boxplot for
    """

    print(f"\nPlotting Boxplot of {on}")

    plt.figure(figsize=BOXPLOT_SIZE)
    plt.boxplot(df[on],
                vert=False,
                widths=0.7,
                flierprops={"markerfacecolor": "black"})

    # * Only show the 5-number summary
    min = df[on].min()
    q1 = df[on].quantile(0.25)
    median = df[on].quantile(0.5)
    q3 = df[on].quantile(0.75)
    max = df[on].max()
    plt.xticks([min, q1, median, q3, max])
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.yticks([])

    plt.title(on, fontsize=TITLE_SIZE)
    plt.tight_layout()

    plt.savefig(DIR_SAVE_BOXPLOT.format(on=on), dpi=PLOT_DPI)
    plt.close()

    print(f"Saved Boxplot of {on}")


def plot_relationships(df):
    """
    @context
        * Creates and saves a pairwise relationship plot with all `MAIN_EFFECTS`
          and the `DEPENDENT_VAR`
        * Plot is a corner plot
            * Scatter plots on the bottom-left corner
            * Density plots along the diagonals

    @parameters
        * df : Dataframe
            * Dataframe containing the `MAIN_EFFECTS` and the `DEPENDENT_VAR`
    """

    print(f"\nPlotting {TITLE_RELATIONSHIPS}")

    vars = MAIN_EFFECTS + [DEPENDENT_VAR]
    n = len(vars)

    figure, axis = plt.subplots(ncols=n, nrows=n, figsize=RELATIONSHIP_SIZE)

    # * Create scatters between each `var` in bottom-left corner of `figure`
    for j in range(n - 1):
        for i in range(j + 1, n):
            axis[i, j].scatter(df[vars[j]], df[vars[i]])

            # * Hide axis numbers of all plots
            # * Axis numbers on the outside plot are later shown
            axis[i, j].get_xaxis().set_visible(False)
            axis[i, j].get_yaxis().set_visible(False)

            # * Make the top-right corner blank (`i` and `j` flipped)
            axis[j, i].axis("off")

    # * Create density plots (histograms) along the diagonals
    for i, var in enumerate(vars):
        axis[i, i].hist(df[var], density=True)

        # * Overlay a KDE over each histogram
        x_min, x_max = axis[i, i].get_xlim()
        kde = gaussian_kde(df[var])
        linespace = np.linspace(x_min, x_max)
        axis[i, i].plot(linespace, kde(linespace))

        # * Hide axis numbers of all plots
        axis[i, i].get_xaxis().set_visible(False)
        axis[i, i].get_yaxis().set_visible(False)

    # * Name and show axis of outside plots
    for i, var in enumerate(vars):
        axis[-1, i].set_xlabel(var, fontsize=LABEL_SIZE)
        axis[-1, i].get_xaxis().set_visible(True)

        # * No y-axis label on top-left density plot
        if i != 0:
            axis[i, 0].set_ylabel(var, fontsize=LABEL_SIZE)
            axis[i, 0].get_yaxis().set_visible(True)

    figure.suptitle(TITLE_RELATIONSHIPS, fontsize=TITLE_SIZE)
    figure.tight_layout()

    figure.savefig(DIR_SAVE_RELATIONSHIPS, dpi=PLOT_DPI)
    plt.close(figure)

    print(f"Saved {TITLE_RELATIONSHIPS}")


def plot_diagnostics(model):
    """
    @context
        * Creates and saves diagnostics plots for `model`

    @parameters
        * model : Model
            * Model to plot diagnostics for
    """

    print(f"\nPlotting {TITLE_DIAGNOSTICS}")

    # * Find measures used in diagnostics
    n = model.nobs
    residuals = model.resid_deviance
    standardised_residuals = model.get_influence().resid_studentized
    sqrt_standardised_residuals = np.sqrt(abs(standardised_residuals))
    fitted_values = model.fittedvalues
    theoretical_quantiles = [norm.ppf(i / (n + 1)) for i in range(1, n + 1)]
    leverage = model.get_influence().hat_matrix_diag

    figure, axis = plt.subplots(ncols=2, nrows=2, figsize=DIAGNOSTIC_SIZE)

    # * Residuals vs. fitted plot
    axis[0, 0].scatter(fitted_values, residuals)
    axis[0, 0].axhline(y=0, color="black", zorder=-10)
    lowess_line(axis[0, 0], fitted_values, residuals)
    axis[0, 0].set_title(TITLE_RESIDUALS_VS_FITTED, fontsize=SUBTILE_SIZE)
    axis[0, 0].set_xlabel(LABEL_X_RESIDUALS_VS_FITTED, fontsize=LABEL_SIZE)
    axis[0, 0].set_ylabel(LABEL_Y_RESIDUALS_VS_FITTED, fontsize=LABEL_SIZE)

    # * Normal Q-Q plot
    axis[0, 1].scatter(theoretical_quantiles, sorted(standardised_residuals))
    axis[0, 1].axline((0, 0), slope=1, color="black", zorder=-10)
    axis[0, 1].set_title(TITLE_NORMAL_QQ, fontsize=SUBTILE_SIZE)
    axis[0, 1].set_xlabel(LABEL_X_NORMAL_QQ, fontsize=LABEL_SIZE)
    axis[0, 1].set_ylabel(LABEL_Y_NORMAL_QQ, fontsize=LABEL_SIZE)

    # * Scale-location plot
    axis[1, 0].scatter(fitted_values, sqrt_standardised_residuals)
    lowess_line(axis[1, 0], fitted_values, sqrt_standardised_residuals)
    axis[1, 0].set_title(TITLE_SCALE_LOCATION, fontsize=SUBTILE_SIZE)
    axis[1, 0].set_xlabel(LABEL_X_SCALE_LOCATION, fontsize=LABEL_SIZE)
    axis[1, 0].set_ylabel(LABEL_Y_SCALE_LOCATION, fontsize=LABEL_SIZE)

    # * Residuals vs. leverage plot
    axis[1, 1].scatter(leverage, standardised_residuals)
    axis[1, 1].axhline(y=0, color="black", zorder=-10)
    lowess_line(axis[1, 1], leverage, standardised_residuals)
    axis[1, 1].set_title(TITLE_RESIDUALS_VS_LEVERAGE, fontsize=SUBTILE_SIZE)
    axis[1, 1].set_xlabel(LABEL_X_RESIDUALS_VS_LEVERAGE, fontsize=LABEL_SIZE)
    axis[1, 1].set_ylabel(LABEL_Y_RESIDUALS_VS_LEVERAGE, fontsize=LABEL_SIZE)

    figure.suptitle(TITLE_DIAGNOSTICS, fontsize=TITLE_SIZE)
    figure.tight_layout()

    figure.savefig(DIR_SAVE_DIAGNOSTICS, dpi=PLOT_DPI)
    plt.close(figure)

    print(f"Saved {TITLE_DIAGNOSTICS}")


def lowess_line(axis, x, y):
    """
    @context
        * Draws a LOWESS line on `axis` by smoothing `x` with `y`

    @parameters
        * axis : Axis
            * Axis to draw line on
        * x : float
            * x-values to smooth
        * y : float
            * y-values to smooth
    """
    smoothed = sm.nonparametric.lowess(y, x, frac=LOWESS_SMOOTHING)
    axis.plot(smoothed[:, 0], smoothed[:, 1], color="red")


if __name__ == "__main__":
    main()
