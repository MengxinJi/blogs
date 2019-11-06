---
layout: post
title: "Making Regression Make Sense"
description: ""
date: 2019-01-11
tags: [Most Harmless Econometrics, Economics]
comments: true
---
In most cases, regression is used with observational data. Without the benefit of random assignment, regression estimates may or may not have a causal interpretation. 

## Regression Fundamentals 


Before discussing about random experiment regression, we first review the following properties about general regression:

(i) the intimate connection between the population regression function and the conditional expectation function;(ii) how and why regression coefficients change as covariates are added or removed from the model;(iii) the close link between regression and other "control strategies" such as matching;
(iv) the sampling distribution of regression estimates.

The question of whether the earnings - schooling relationship is causal is of enormous importance. Even without resolving the difficult question of causality, however, it is clear that education predicts earnings in a narrow statistical sense. This predictive power is compellingly summarized by the conditional expectation function (CEF).

Expectation is a population concept. In practice, data usually come in the form of samples and rarely consist of an entire population. We therefore use samples to make inferences about the population. 

(Properties of CEF can be found in Chapter 3)

## Regression and Causality

A regression is causal when the CEF it approximates is causal. The CEF is causal when it describes differences in average potential outcomes for a fixed reference population.

As we discussed in previous post, experiments ensure that the causal variable of interest is independent of potential outcomes so that the groups being compared are truly comparable. Here, we would like to generalize this notion to causal variables that take on more than two values, and to more complicated situations where we must hold a variety of "control variables" fixed for causal inferences to be valid. This leads to the conditional independence assumption (CIA), a core assumption that provides the (sometimes implicit) justification for the causal interpretation of regression. This assumption is sometimes called selection-on-observables because the covariates to be held fixed are assumed to be known and observed. The CIA asserts that conditional on observed characteristics, Xi, selection bias disappears.

The omitted variables bias (OVB) formula describes the relationship between regression estimates in models with different sets of control variables. This important formula is often motivated by the notion that a longer regression, i.e., one with more controls, has a causal interpretation, while a shorter regression does not.s determined.

We have made the point that control for covariates can make the CIA more plausible. But more control is not always better. Some variables are bad controls and should not be included in a regression model even when their inclusion might be expected to change the short regression coefficients. Bad controls are variables that are themselves outcome variables in the notional experiment at hand. That is, bad controls might just as well be dependent variables too. Good controls are variables that we can think of as having been fixed at the time the regressor of interest was determined.



















