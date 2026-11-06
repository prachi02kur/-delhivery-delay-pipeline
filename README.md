# Delhivery Last-Mile Delay Intelligence Pipeline

This is a Capstone Project for Artificial Intelligence and Machine Learning Certification Program, E&ICT Academy, IIT Roorkee. The data used in the project is synthetic. 
This project consist of three parts, Part A is for cleaning the data, Part B is used to determine how long will a parcel take, Part C is for which parcels will miss their promise and Part D is for determining which hubs need which playbook. All the library used are free.

## How to run

List the commands in dependency order, from the repo root:

```bash
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```
The scripts must be run in this order because each one reads files written by an earlier one. `package_model.py` needs `part_c_final_config.csv`, and `predict_delay.py` needs `hub_segments.csv` and the models/ files, so this order matters.

## Data findings

**Data preparation.** The raw export has 1,224 rows. After removing 24 exact
duplicates, 1,200 unique shipments remain. The table had 36 missing distances,
24 missing weights, 12 missing declared values and 18 missing payment modes
(counted before filling), plus 5 parcels with a scanner-fault weight above 100 kg,
which were treated as missing. `service_type` came in 12 different spellings and was
reduced to 3. After cleaning there are no missing values in the 17 features, the
target or `is_delayed`. Overall 278 of the 1,200 parcels are delayed.
All numbers below are copied from `outputs/part_a_checks.csv`,
`outputs/part_a_delay_by_group.csv` and `outputs/part_a_correlations.csv`.

**(i) Express parcels are delayed most often.** The delay rate is 0.4897 for Express
(119 of 243 parcels), 0.2056 for Standard (132 of 642) and 0.0857 for Economy
(27 of 315). Express parcels average 36.8056 hours against a 36-hour promise, so
there is almost no slack. Economy parcels average 82.0931 hours against 96.

**(ii) Delay rate rises sharply from Tier 1 to Tier 3.** It is 0.04 for Tier 1
(16 of 400), 0.215 for Tier 2 (86 of 400) and 0.44 for Tier 3 (176 of 400).
Mean delivery hours rise in step: 48.5664, 58.8194 and 63.4943.

**(iii) Delay rate rises steeply with distance.** It is 0.0219 for 0-200 km,
0.1368 for 201-400 km, 0.4364 for 401-600 km and 0.8523 for 601+ km (75 of 88
parcels). Most parcels travelling more than 600 km miss their promise.

**(iv) Strongest correlations with `delivery_hours`.** Ranking the columns of
`part_a_correlations.csv` by absolute correlation, with `delivery_hours` itself
excluded, the top three are `promised_hours` (0.8372), `distance_km` (0.4013) and
`n_stops` (0.3247). `promised_hours` is highest because the service type sets the
base delivery time. That does not mean longer promises are delayed more often: Express
has the shortest promise and the highest delay rate, because a delay compares the
hours with the promise.

**(v) Why `declared_value_inr` was log-transformed.** On the median-filled column the
mean is INR 1,533.05 (1533.0542 in the file) against a median of INR 1,121.50, which
shows a long right tail. After the natural log the mean (7.0289) and the median
(7.0233) are almost equal, so the transformed column is roughly symmetric.

**Delivery hours outliers.** The quartiles of `delivery_hours` are 43.2550 and 72.5625,
and the 1.5 × IQR bounds are -0.7063 and 116.5238. No parcel falls below the lower bound
and 1 parcel lies above the upper bound.

### Model results (regression)

**Setup.** The 1,200 shipments were split 70/15/15 with `random_state=42` into 840
training, 180 validation and 180 test rows. The `StandardScaler` was fitted on the 840
training rows only. In `outputs/part_b_checks.csv` the largest absolute column mean of the
scaled training rows is 0.0 (zero up to floating-point error), while the scaled test rows
give 0.2187. A non-zero test value is what we expect when the scaler has learned its
statistics from the training rows alone.

**Baseline.** Predicting the training mean for every validation row gives MAE 15.7952,
RMSE 18.8552 and R² -0.0212. An R² of about zero means it explains none of the variation
in delivery time, so every real model has to beat it.

**Tree depth (`part_b_tree_depth.csv`).** At depth 1 the training RMSE is 11.6334 and the
validation RMSE is 12.1324. Both are high and only 0.4990 apart, so the tree underfits
(high bias). With no depth limit the training RMSE is 0.0000 but the validation RMSE is
7.8368, a gap of 7.8368 hours, so the tree memorises the training rows (high variance).
The validation RMSE is lowest at depth 6 (6.2796, with a train-validation gap of 1.5682),
so depth 6 is the decision-tree depth used in the model comparison. Depth 6 is 5.8528
hours better than depth 1 on validation RMSE.

**Four candidate models (`part_b_metrics.csv`, validation rows).** Linear regression has
MAE 3.9222, RMSE 4.8912 and R² 0.9313. Gradient boosting is almost identical, with
MAE 3.9166, RMSE 4.9165 and R² 0.9306. The random forest has MAE 4.2882, RMSE 5.5095 and
R² 0.9128, and the depth-6 decision tree has MAE 5.0971, RMSE 6.2796 and R² 0.8867. All
four beat the baseline by a wide margin. Linear regression and gradient boosting differ
by only 0.0253 hours of RMSE, so they are practically tied, and linear regression is
lower, which makes it the final model under the lowest-validation-RMSE rule.

**Learning rate (`part_b_gb_learning_rate.csv`).** With 100 trees, a learning rate of
0.01 underfits (validation R² 0.7127), because each tree corrects only a small part of
the remaining error and 100 trees are not enough to catch up. A rate of 1.0 is too
aggressive (validation R² 0.8440), because each tree's correction is applied in full and
the ensemble overshoots. A rate of 0.1 is best, with validation R² 0.9306 and validation
RMSE 4.9165.

**Cross-validation (`part_b_cv.csv`).** The two best candidates were compared with the
same 5-fold `KFold(n_splits=5, shuffle=True, random_state=42)` on the 1,020 training and
validation rows. Linear regression has a mean R² of 0.9261 (std 0.0048) and gradient
boosting has 0.9353 (std 0.0045). Cross-validation therefore favours gradient boosting
slightly, the opposite of the validation ranking. The difference is 0.0092 in R², about
two standard deviations of either model's fold scores, so the two models are practically
equivalent. The final model was chosen on validation RMSE as the brief requires. A
five-fold estimate on 1,020 rows is less affected by which 180 rows happened to be in
the validation set, but it is still a small sample. If a straight line has no way to
represent interactions between features, that could explain why boosting does a little
better here, but I have not tested that in this repository, so it is a possible
explanation and not a finding.

**Final test score.** Linear regression, the lowest-validation-RMSE model, was scored
once on the 180 test rows: MAE 3.5671, RMSE 4.5236 and R² 0.9493. This R² is higher than
its validation R² (0.9313). The two sets are different samples of 180 rows, so the gap
is sampling variation and not evidence of leakage or of a better model than validation
suggested. No other model was scored on the test rows.

## Model results (classification)


**Split and baseline.** The stratified split gives 840 training, 180 validation and 180 test
rows, with 195, 42 and 41 delayed parcels. Predicting "not delayed" for every parcel
scores an accuracy of 0.7667 on validation but a recall of 0: it finds none of the 42
delayed parcels. It is useless because the goal is to flag parcels before they are late,
and accuracy hides this when about 23% of parcels are delayed. At the 0.5 rule, logistic
regression (F1 0.8046), the depth-6 tree (0.7470) and the default forest (0.7848) all
catch delayed parcels (`part_c_models.csv`).

**Signal and selection.** All five bookkeeping columns have F-scores between 0.0230 and
4.5847 and F-ranks of 8 or lower, so none is in the top 7 by F-score. Their combined
share of the forest's importance is still 0.2245, because trees can split on random
values, so the F-test is the stronger evidence of "no signal". Validation F1 for
k = 5, 7, 10, 12 and 17 is 0.6988, 0.8333, 0.7857, 0.8049 and 0.7848
(`part_c_selectk.csv`). k = 7 is highest, but the gap to k = 12 and k = 17 is only a few
parcels among 42 delayed ones.

**Tuning.** The untuned forest on the 7 columns has a validation F1 of 0.8333, higher
than the grid search (0.7750) and the randomized search (0.7848), so it was chosen
(`part_c_tuning.csv`). With 42 delayed validation parcels, one parcel moves recall by
about 0.024, so these differences are within noise, and tuning on a 180-row validation
set can fit that set instead of the problem.

**Threshold.** A missed delayed parcel costs more than an unnecessary call, so the
threshold is the highest one with validation recall of at least 0.80. Recall is 0.8333 at
0.5 and 0.7619 at 0.6, so the threshold is 0.5, with 9 false positives, 7 false negatives
and precision 0.7955 (`part_c_thresholds.csv`). The F1 at 0.5 is 0.8140 here but 0.8333
in the tuning file, because `predict` labels a probability of exactly 0.5 as not delayed
while the threshold rule flags it, which adds 2 false positives. Nothing was changed
because of this.

**Test.** Scored once on the 180 test rows at 0.5, the final model has precision 0.6889,
recall 0.7561 and F1 0.7209 (`part_c_final_test.csv`): about 7 of every 10 flagged parcels
are really delayed, and about 7.6 of every 10 delayed parcels are caught. Test recall is
below the validation recall (0.8333) and below the 0.80 target. With only 41 delayed
test parcels the gap is about 3 parcels. The model and threshold were chosen on the
validation rows and not changed after seeing the test result, so the validation figures
are likely optimistic.

## Operations recommendation

**Where to focus: the nine Escalate hubs.** K-Means on the four hub-level profile
measures puts nine hubs in the Escalate segment, with a segment delay rate of 0.4600
(`outputs/segment_profile.csv`). In `outputs/hub_profile.csv` their delay rates are
Guwahati (HUB19) 0.62, Agartala (HUB24) 0.54, Raipur (HUB20) 0.48, Nagpur (HUB13) 0.46,
Patna (HUB17) 0.46, Siliguri (HUB23) 0.44, Bhubaneswar (HUB21) 0.40, Ranchi (HUB18) 0.38
and Kochi (HUB15) 0.36. These hubs average 470.4757 km per lane and 84.3444 stops per
route, against 188.9296 km and 55.3686 stops in the Standard segment, whose delay rate is
0.0257. The Watch segment (eight hubs, delay rate 0.1550) sits in between. Operations
should run the Escalate playbook at all nine hubs, the Watch playbook at the eight Watch
hubs and the Standard playbook at the seven Standard hubs.

**Four hubs where the tier label is misleading
(`outputs/tier_vs_segment_mismatch.csv`).** For these four hubs the segment, not the
tier label, should choose the playbook.
- Kolkata (HUB07) is labelled Tier 1 but behaves like a Watch hub: 304.736 km, 70.76
  stops and a delay rate of 0.14, while the other seven Tier 1 hubs have delay rates of
  0.00 to 0.04. The tier label understates its risk.
- Nagpur (HUB13) and Kochi (HUB15) are labelled Tier 2 but behave like Escalate hubs:
  473.320 and 466.514 km, 83.92 and 84.10 stops, and delay rates of 0.46 and 0.36,
  while the other six Tier 2 hubs have delay rates of 0.08 to 0.24. The tier label
  understates their risk.
- Dehradun (HUB22) is labelled Tier 3 but behaves like a Watch hub: 316.738 km,
  73.28 stops and a delay rate of 0.20, while the other seven Tier 3 hubs have delay
  rates of 0.38 to 0.62. The tier label overstates its risk, and sending it the full
  Escalate playbook would spend manager time on a hub that behaves like a regional one.

**Alert threshold and what a flag means.** Parcels are flagged when the predicted
delay probability is at least 0.5. This is the highest threshold at which validation
recall is still at least 0.80 (0.8333 at 0.5, 0.7619 at 0.6), chosen because a missed
delayed parcel costs more than an unnecessary call. On the 180 test rows
(`outputs/part_c_final_test.csv`) the precision is 0.6889, so about 7 of every 10 flagged
parcels are really delayed and about 3 are unnecessary calls. The recall is 0.7561, so
about 7.6 of every 10 delayed parcels are caught and about 2.4 are missed. Recall on the
test rows is below the 0.80 target, so the head should expect to miss somewhat more
delayed parcels than the validation figures suggest, and should treat the threshold as a
setting to review once real outcomes arrive.

**The 48 new parcels (`outputs/new_shipment_actions.csv`).** 8 of the 48 are flagged.
Six of those eight sit in Escalate-segment hubs and need an escalation to the hub
manager. The other two get a proactive customer call. The remaining 40 parcels need no
action.

**Why K = 3 is kept.** Operations runs exactly three playbooks, so K = 3 is fixed by the
business, and the separation evidence does not contradict it. The across-to-within
distance ratio is 5.3928 for K = 3 and keeps rising, to 6.3047 for K = 4 and 6.8866 for
K = 5 (`outputs/part_d_separation.csv`). But the K = 4 solution has cluster sizes 8-7-5-4
and the K = 5 solution has 8-7-4-3-2, so each contains a cluster of fewer than six hubs,
which is too small to carry its own playbook. The smallest K = 3 cluster has seven hubs,
and the ratio rises with K partly because smaller clusters are tighter, not because they
are more useful.


## Production plan

**(a) Production-readiness, mapped to the repository.**
- *Accuracy:* held-out test metrics are in `part_b_metrics.csv` (linear regression R² 0.9493) and `part_c_final_test.csv` (precision 0.6889, recall 0.7561, F1 0.7209), and the test rows were scored once.
- *Reliability:* `clean_shipments` handles duplicates, gaps, mixed spellings and the scanner-fault weight. The scripts assert row counts and no missing values, and `part_d_reload_check.csv` shows the reloaded model gives identical probabilities (difference 0.0).
- *Reproducibility:* pinned `random_state` values, `requirements.txt` with exact versions, `generate_data.py`, and a fixed run order in this README. `predict_delay.py` gives byte-identical output on repeated runs.
- *Maintainability:* the cleaning and feature code lives once in `src/shipment_prep.py`. The threshold, feature order, medians and segments are read from files, not retyped.
- *Consistency:* training and scoring both call the same `clean_shipments` and `build_features`. `feature_order.csv` fixes the column order and `impute_values.csv` stores the historical medians that new parcels are filled with.

**(b) What to monitor, with triggers.** The triggers are starting values to recalibrate once real weekly volumes are known.
- Weekly delayed share of delivered parcels: the history is 0.2317 (278 of 1,200). Review if it leaves the range 0.15 to 0.32 for two weeks in a row.
- Weekly share of parcels flagged: 0.25 on the test rows (45 of 180). Review if it moves more than 0.10 from that for two weeks, because the inputs or the model's behaviour may have shifted.
- Recall on parcels whose outcome is now known, over a rolling four weeks: validation recall was 0.8333 and test recall 0.7561. Retrain if it falls below 0.70, and rerun the threshold rule if it stays under 0.80.
- Share of parcels filled by a median, and the number of weights above 100 kg: a jump means the data feed has changed.

**(c) Retraining, the sale window and drift.** Retrain monthly by appending delivered parcels, then rerun the pipeline and `package_model.py`. Retrain at once if the recall trigger fires. Re-choose the threshold with the same rule (highest threshold with validation recall of at least 0.80), and re-run the K-Means segments each quarter to check whether the four mismatched hubs have moved.

The 15 to 28 September sale window (`sale_event_flag`) should be compared with the same kind of event, such as the previous sale, and not with ordinary weeks. A spike in delays or flags during a sale should not trigger retraining on its own. The history contains a single sale, so the model has seen only one example. After each sale, add its outcomes to the training data and check recall within the window separately.

Data drift here would look like a change in mean distance or stops per hub, a shift in the service mix or the rain share, new hubs or a hub changing its operating pattern, a rising share of imputed values, or a gap opening between the flagged share and the delayed share.

## Limitation

- **The data is synthetic.** Every row comes from `generate_data.py`, so the results show that the pipeline works, not how well it would perform on real Delhivery parcels. Real data would bring effects this dataset does not contain, and the model's accuracy, the segments and the threshold would all need re-checking.
- **The delayed counts are small.** The test set has 41 delayed parcels and the validation set has 42, so one parcel moves recall by about 0.024. Test recall (0.7561) is below validation recall (0.8333) and below the 0.80 target, and the gap is about three parcels. The test figures therefore carry wide uncertainty, and I would not read them to the second decimal.
- **Choices were made on one validation split.** The feature count (k = 7), the model configuration and the 0.5 threshold were all chosen on the same 180 validation rows, so the validation figures are probably optimistic. The three tuning results (0.8333, 0.7848 and 0.7750 validation F1) are close enough that a different split could have changed the winner. The same applies in Part B, where linear regression and gradient boosting are practically tied.
- **The threshold rule is a stated rule, not a measured cost.** The brief says a missed delayed parcel costs more than an unnecessary call, but no cost in INR is given. The "recall of at least 0.80" rule is therefore a judgement about the trade-off and not the result of minimising a cost. If the real costs were known, the threshold could be set to minimise them.
- **One sale window and a short history.** The data covers about 90 days and contains a single sale (15 to 28 September), so the model has seen one example of that effect and no seasonal pattern. Behaviour in other seasons, festivals or monsoon months is untested.
- The packaged classifier is fitted on the 840 training rows only (not training plus validation), and the pickle depends on the exact scikit-learn version.
