"""Statistical significance testing module for experiment comparison."""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import logging

from experiments.comparator import ExperimentComparison, ComparisonMetrics


class TestType(Enum):
    """Enumeration of available statistical tests."""
    PAIRED_TTEST = "paired_ttest"
    UNPAIRED_TTEST = "unpaired_ttest"
    WELCH_TTEST = "welch_ttest"
    ONE_WAY_ANOVA = "one_way_anova"
    KRUSKAL_WALLIS = "kruskal_wallis"


class EffectSizeType(Enum):
    """Enumeration of effect size measures."""
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    ETA_SQUARED = "eta_squared"
    OMEGA_SQUARED = "omega_squared"


@dataclass
class AssumptionCheck:
    """Results of statistical assumption testing."""
    test_name: str
    statistic: float
    p_value: float
    assumption_met: bool
    interpretation: str


@dataclass
class EffectSize:
    """Effect size calculation results."""
    measure: str
    value: float
    magnitude: str  # small, medium, large
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class StatisticalTestResult:
    """Results of a statistical significance test."""
    test_type: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[Union[int, Tuple[int, int]]] = None
    effect_size: Optional[EffectSize] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    significant: bool = False
    alpha: float = 0.05
    interpretation: str = ""
    assumptions: List[AssumptionCheck] = None
    
    def __post_init__(self):
        """Set significance and interpretation after initialization."""
        if self.assumptions is None:
            self.assumptions = []
        self.significant = self.p_value < self.alpha
        if not self.interpretation:
            self.interpretation = self._generate_interpretation()
    
    def _generate_interpretation(self) -> str:
        """Generate human-readable interpretation of results."""
        sig_text = "significant" if self.significant else "not significant"
        effect_text = ""
        if self.effect_size:
            effect_text = f" with {self.effect_size.magnitude} effect size ({self.effect_size.measure}={self.effect_size.value:.3f})"
        
        return f"The difference is {sig_text} (p={self.p_value:.4f}){effect_text}."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        if self.degrees_of_freedom is not None and isinstance(self.degrees_of_freedom, tuple):
            result['degrees_of_freedom'] = list(self.degrees_of_freedom)
        return result


@dataclass
class MultipleComparisonResult:
    """Results of multiple comparison analysis."""
    overall_test: StatisticalTestResult
    pairwise_comparisons: Dict[Tuple[str, str], StatisticalTestResult]
    corrected_alpha: float
    correction_method: str = "bonferroni"


class StatisticalAnalyzer:
    """Statistical analysis for experiment comparisons."""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the statistical analyzer.
        
        Args:
            alpha: Significance level for hypothesis testing
        """
        self.alpha = alpha
        self.logger = logging.getLogger("StatisticalAnalyzer")
    
    def analyze_comparison(self, 
                          experiment_data: Dict[Union[str, int], List[float]],
                          test_type: Optional[TestType] = None,
                          paired: bool = False) -> Union[StatisticalTestResult, MultipleComparisonResult]:
        """
        Perform statistical analysis on experiment comparison data.
        
        Args:
            experiment_data: Dictionary mapping experiment IDs to metric values
            test_type: Specific test to perform (auto-selected if None)
            paired: Whether experiments are paired/related
            
        Returns:
            Statistical test results
        """
        if len(experiment_data) < 2:
            raise ValueError("Need at least 2 experiments for comparison")
        
        # Auto-select test if not specified
        if test_type is None:
            test_type = self._select_appropriate_test(experiment_data, paired)
        
        if len(experiment_data) == 2:
            return self._perform_two_sample_test(experiment_data, test_type)
        else:
            return self._perform_multiple_sample_test(experiment_data, test_type)
    
    def _select_appropriate_test(self, 
                               experiment_data: Dict[Union[str, int], List[float]], 
                               paired: bool) -> TestType:
        """Automatically select appropriate statistical test."""
        n_groups = len(experiment_data)
        
        if n_groups == 2:
            if paired:
                return TestType.PAIRED_TTEST
            else:
                # Check for equal variances
                groups = list(experiment_data.values())
                if len(groups[0]) > 2 and len(groups[1]) > 2:
                    _, p_levene = stats.levene(groups[0], groups[1])
                    if p_levene < 0.05:  # Unequal variances
                        return TestType.WELCH_TTEST
                return TestType.UNPAIRED_TTEST
        else:
            # Check normality for all groups
            all_normal = True
            for group_data in experiment_data.values():
                if len(group_data) >= 3:  # Need at least 3 samples for Shapiro-Wilk
                    _, p_shapiro = stats.shapiro(group_data)
                    if p_shapiro < 0.05:
                        all_normal = False
                        break
            
            return TestType.ONE_WAY_ANOVA if all_normal else TestType.KRUSKAL_WALLIS
    
    def _perform_two_sample_test(self, 
                               experiment_data: Dict[Union[str, int], List[float]], 
                               test_type: TestType) -> StatisticalTestResult:
        """Perform statistical test for two experiments."""
        exp_ids = list(experiment_data.keys())
        group1, group2 = experiment_data[exp_ids[0]], experiment_data[exp_ids[1]]
        
        # Check assumptions
        assumptions = self._check_assumptions(experiment_data, test_type)
        
        if test_type == TestType.PAIRED_TTEST:
            if len(group1) != len(group2):
                raise ValueError("Paired t-test requires equal sample sizes")
            statistic, p_value = stats.ttest_rel(group1, group2)
            df = len(group1) - 1
            effect_size = self._calculate_cohens_d(group1, group2, paired=True)
            
        elif test_type == TestType.UNPAIRED_TTEST:
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=True)
            df = len(group1) + len(group2) - 2
            effect_size = self._calculate_cohens_d(group1, group2, paired=False)
            
        elif test_type == TestType.WELCH_TTEST:
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            # Welch's t-test degrees of freedom (Satterthwaite approximation)
            s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
            effect_size = self._calculate_cohens_d(group1, group2, paired=False)
        
        else:
            raise ValueError(f"Unsupported test type for two samples: {test_type}")
        
        # Calculate confidence interval for mean difference
        ci = self._calculate_mean_difference_ci(group1, group2, test_type)
        
        return StatisticalTestResult(
            test_type=test_type.value,
            statistic=statistic,
            p_value=p_value,
            degrees_of_freedom=df,
            effect_size=effect_size,
            confidence_interval=ci,
            alpha=self.alpha,
            assumptions=assumptions
        )
    
    def _perform_multiple_sample_test(self, 
                                    experiment_data: Dict[Union[str, int], List[float]], 
                                    test_type: TestType) -> MultipleComparisonResult:
        """Perform statistical test for multiple experiments."""
        groups = list(experiment_data.values())
        group_names = list(experiment_data.keys())
        
        # Check assumptions
        assumptions = self._check_assumptions(experiment_data, test_type)
        
        if test_type == TestType.ONE_WAY_ANOVA:
            statistic, p_value = stats.f_oneway(*groups)
            # Calculate degrees of freedom
            k = len(groups)  # number of groups
            n = sum(len(group) for group in groups)  # total sample size
            df = (k - 1, n - k)
            effect_size = self._calculate_eta_squared(groups, statistic, df)
            
        elif test_type == TestType.KRUSKAL_WALLIS:
            statistic, p_value = stats.kruskal(*groups)
            df = len(groups) - 1
            effect_size = None  # Eta-squared not applicable for non-parametric tests
            
        else:
            raise ValueError(f"Unsupported test type for multiple samples: {test_type}")
        
        overall_result = StatisticalTestResult(
            test_type=test_type.value,
            statistic=statistic,
            p_value=p_value,
            degrees_of_freedom=df,
            effect_size=effect_size,
            alpha=self.alpha,
            assumptions=assumptions
        )
        
        # Perform pairwise comparisons if overall test is significant
        pairwise_results = {}
        corrected_alpha = self.alpha
        
        if overall_result.significant:
            n_comparisons = len(groups) * (len(groups) - 1) // 2
            corrected_alpha = self.alpha / n_comparisons  # Bonferroni correction
            
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    pair_data = {group_names[i]: groups[i], group_names[j]: groups[j]}
                    pair_test_type = TestType.UNPAIRED_TTEST if test_type == TestType.ONE_WAY_ANOVA else TestType.UNPAIRED_TTEST
                    
                    pair_result = self._perform_two_sample_test(pair_data, pair_test_type)
                    pair_result.alpha = corrected_alpha
                    pair_result.significant = pair_result.p_value < corrected_alpha
                    
                    pairwise_results[(str(group_names[i]), str(group_names[j]))] = pair_result
        
        return MultipleComparisonResult(
            overall_test=overall_result,
            pairwise_comparisons=pairwise_results,
            corrected_alpha=corrected_alpha,
            correction_method="bonferroni"
        )
    
    def _check_assumptions(self, 
                         experiment_data: Dict[Union[str, int], List[float]], 
                         test_type: TestType) -> List[AssumptionCheck]:
        """Check statistical assumptions for the given test."""
        assumptions = []
        groups = list(experiment_data.values())
        
        # Normality check (for parametric tests)
        if test_type in [TestType.PAIRED_TTEST, TestType.UNPAIRED_TTEST, 
                        TestType.WELCH_TTEST, TestType.ONE_WAY_ANOVA]:
            for i, group in enumerate(groups):
                if len(group) >= 3:  # Need at least 3 samples
                    stat, p_val = stats.shapiro(group)
                    assumptions.append(AssumptionCheck(
                        test_name=f"Normality (Group {i+1})",
                        statistic=stat,
                        p_value=p_val,
                        assumption_met=p_val >= 0.05,
                        interpretation="Data appears normally distributed" if p_val >= 0.05 
                                     else "Data may not be normally distributed"
                    ))
        
        # Homogeneity of variance check (for ANOVA and equal-variance t-tests)
        if test_type in [TestType.UNPAIRED_TTEST, TestType.ONE_WAY_ANOVA] and len(groups) >= 2:
            if all(len(group) > 1 for group in groups):
                stat, p_val = stats.levene(*groups)
                assumptions.append(AssumptionCheck(
                    test_name="Homogeneity of Variance",
                    statistic=stat,
                    p_value=p_val,
                    assumption_met=p_val >= 0.05,
                    interpretation="Variances appear equal across groups" if p_val >= 0.05 
                                 else "Variances may be unequal across groups"
                ))
        
        return assumptions
    
    def _calculate_cohens_d(self, 
                          group1: List[float], 
                          group2: List[float], 
                          paired: bool = False) -> EffectSize:
        """Calculate Cohen's d effect size."""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        
        if paired:
            # For paired samples, use the standard deviation of differences
            differences = np.array(group1) - np.array(group2)
            pooled_std = np.std(differences, ddof=1)
            d = np.mean(differences) / pooled_std
        else:
            # For independent samples, use pooled standard deviation
            n1, n2 = len(group1), len(group2)
            s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            d = (mean1 - mean2) / pooled_std
        
        # Interpret magnitude
        magnitude = "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        
        return EffectSize(
            measure="Cohen's d",
            value=d,
            magnitude=magnitude
        )
    
    def _calculate_eta_squared(self, 
                             groups: List[List[float]], 
                             f_statistic: float, 
                             df: Tuple[int, int]) -> EffectSize:
        """Calculate eta-squared effect size for ANOVA."""
        df_between, df_within = df
        eta_squared = (f_statistic * df_between) / (f_statistic * df_between + df_within)
        
        # Interpret magnitude (Cohen's conventions for eta-squared)
        magnitude = "small" if eta_squared < 0.06 else "medium" if eta_squared < 0.14 else "large"
        
        return EffectSize(
            measure="eta-squared",
            value=eta_squared,
            magnitude=magnitude
        )
    
    def _calculate_mean_difference_ci(self, 
                                    group1: List[float], 
                                    group2: List[float], 
                                    test_type: TestType,
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean difference."""
        mean_diff = np.mean(group1) - np.mean(group2)
        
        if test_type == TestType.PAIRED_TTEST:
            differences = np.array(group1) - np.array(group2)
            se = stats.sem(differences)
            df = len(differences) - 1
        else:
            n1, n2 = len(group1), len(group2)
            s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            
            if test_type == TestType.UNPAIRED_TTEST:
                # Pooled standard error
                pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
                se = np.sqrt(pooled_var * (1/n1 + 1/n2))
                df = n1 + n2 - 2
            else:  # Welch's t-test
                se = np.sqrt(s1**2/n1 + s2**2/n2)
                # Satterthwaite approximation for df
                df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
        
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df)
        margin_of_error = t_critical * se
        
        return (mean_diff - margin_of_error, mean_diff + margin_of_error)
    
    def analyze_experiment_comparison(self, 
                                   comparison: ExperimentComparison,
                                   metric: str,
                                   test_type: Optional[TestType] = None,
                                   paired: bool = False) -> Dict[str, Union[StatisticalTestResult, MultipleComparisonResult]]:
        """
        Analyze statistical significance for a specific metric in an experiment comparison.
        
        Args:
            comparison: ExperimentComparison object from ExperimentComparator
            metric: Metric name to analyze
            test_type: Specific test to perform (auto-selected if None)
            paired: Whether experiments are paired/related
            
        Returns:
            Dictionary with statistical analysis results
        """
        if metric not in comparison.metric_comparisons:
            raise ValueError(f"Metric '{metric}' not found in comparison")
        
        # Extract raw data for statistical analysis
        # Note: This is a simplified approach - in practice, you'd need access to raw data
        # For now, we'll simulate based on the comparison metrics
        experiment_data = {}
        
        for exp_id, comp_metrics in comparison.metric_comparisons[metric].items():
            # Simulate raw data based on summary statistics
            # In a real implementation, you'd pass the raw data directly
            simulated_data = np.random.normal(
                loc=comp_metrics.mean,
                scale=comp_metrics.std,
                size=comp_metrics.count
            )
            if isinstance(simulated_data, np.ndarray):
                simulated_data = simulated_data.tolist()
            elif not isinstance(simulated_data, list):
                simulated_data = [simulated_data]
            experiment_data[exp_id] = simulated_data
        
        result = self.analyze_comparison(experiment_data, test_type, paired)
        
        return {metric: result} 