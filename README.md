<h1 align="center">Predictive Compensation Engine: Employee Salary Estimation 💼</h1>
<h4 align="center">End-to-End Supervised Regression, Feature Engineering, & Predictive Analytics | IBM SkillsBuild</h4>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/IBM-SkillsBuild-052FAD?style=for-the-badge&logo=ibm&logoColor=white" alt="IBM SkillsBuild"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
    <img src="https://img.shields.io/badge/IBM-SkillsBuild-052FAD?style=for-the-bad
</p>

---

<details open>
  <summary><b>📑 DIRECTORY TERMINAL (TABLE OF CONTENTS)</b></summary>
  <ol>
    <li><a href="#overview">Executive Data Science Overview</a></li>
    <li><a href="#schema">Dataset Architecture & Preprocessing Pipeline</a></li>
    <li><a href="#math">Mathematical Formulations & Regularization</a></li>
    <li><a href="#evaluation">Comparative Model Evaluation Matrix</a></li>
    <li><a href="#importance">Feature Importance & Variance Attribution</a></li>
    <li><a href="#architecture">Repository Architecture</a></li>
    <li><a href="#deployment">Execution & Reproducibility SOP</a></li>
    <li><a href="#certification">Certification & Program Attribution</a></li>
  </ol>
</details>

---

### <a id="overview"></a>🌐 EXECUTIVE DATA SCIENCE OVERVIEW

<div align="justify">
Organizational compensation modeling frequently struggles with structural wage disparities, subjective performance appraisals, and unstandardized market adjustments. Unsystematic salary allocation leads to attrition of mission-critical technical talent and operational budget inefficiencies. 

Developed under the <b>IBM SkillsBuild</b> technical curriculum, this project delivers an end-to-end predictive machine learning framework engineered to accurately estimate employee compensation trajectories based on verifiable workforce parameters. The pipeline ingests multi-dimensional human resource records, executes rigorous exploratory data analysis (EDA), mitigates multi-collinearity, and benchmarks parametric versus non-parametric regression architectures to establish high-fidelity, generalizable salary predictions.
</div>

---

### <a id="schema"></a>📊 DATASET ARCHITECTURE & PREPROCESSING PIPELINE

<div align="justify">
The data engineering pipeline guarantees numeric stability, handles non-linear target interactions, and eliminates covariate leakage:
</div>

* **Data Imputation & Integrity:** Missing continuous records are handled via median-based imputation to suppress skewness, while categorical entries are imputed via modal substitution.
* **Categorical Encoding:** Nominal features (e.g., Department, Education Discipline) are mapped using One-Hot Encoding (`OneHotEncoder(drop='first')`) to prevent the dummy variable trap. Hierarchical attributes (e.g., Job Level, Education Tier) undergo deterministic Ordinal Encoding.
* **Multicollinearity Attenuation:** Predictor interdependencies are measured via the Variance Inflation Factor ($\text{VIF}$). Variables yielding $\text{VIF} > 5.0$ are iteratively regularized or dropped to prevent coefficient inflation in linear models.
* **Feature Normalization:** Continuous numerical covariates are normalized via `StandardScaler` to ensure zero mean ($\mu = 0$) and unit variance ($\sigma^2 = 1$).

| Feature Name | Type | Processing Transformation | Target Significance |
| :--- | :--- | :--- | :--- |
| `YearsExperience` | Continuous | Robust Scaling / Log Transformation | Primary baseline tenure driver |
| `JobLevel` | Ordinal | Ranked Integer Mapping ($1\text{--}5$) | Internal seniority hierarchy |
| `EducationLevel` | Ordinal | Categorical Hierarchy Mapping | Minimum credential threshold |
| `Department` | Nominal | One-Hot Encoding (OHE) | Functional operational sector |
| `PerformanceRating` | Discrete | Mean-Centering Scale | Direct merit-based modifier |
| `Salary` (Target) | Continuous ($y$) | Gaussian Box-Cox / Log Transformation | Continuous annual compensation |

---

### <a id="math"></a>🧪 MATHEMATICAL FORMULATIONS & REGULARIZATION

#### 1. Multiple Linear Regression Formulation
<div align="justify">
The baseline relationship between employee feature matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ and compensation target $\mathbf{y} \in \mathbb{R}^n$ is modeled as:
</div>

$$\hat{y}_i = \beta_0 + \sum_{j=1}^{p} \beta_j X_{ij} + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

#### 2. Ridge ($L_2$) and Lasso ($L_1$) Penalization
<div align="justify">
To prevent overfitting on highly collinear parameters (such as job tier correlated with years in role), regularized loss objective functions are minimized:
</div>

$$\mathcal{L}_{\text{Ridge}}(\boldsymbol{\beta}) = \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$

$$\mathcal{L}_{\text{Lasso}}(\boldsymbol{\beta}) = \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \alpha \sum_{j=1}^{p} \vert{}\beta_j\vert{}$$

#### 3. Error Metrics
<div align="justify">
Performance validation across 5-fold cross-validation folds is calculated via Root Mean Squared Error ($\text{RMSE}$) and Coefficient of Determination ($R^2$):
</div>

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}, \qquad R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

---

### <a id="evaluation"></a>📈 COMPARATIVE MODEL EVALUATION MATRIX

<div align="justify">
All architectures were validated using an 80/20 train-test split cross-validated across 5 folds with hyperparameter grid optimization:
</div>

| Model Architecture | Train $R^2$ | Test $R^2$ | MAE (\$) | RMSE (\$) | Generalization Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ordinary Least Squares (OLS)** | $0.812$ | $0.804$ | $6,420$ | $8,110$ | Baseline Underfitting |
| **Ridge Regression ($\alpha=1.5$)** | $0.835$ | $0.828$ | $5,890$ | $7,450$ | Stable Regularization |
| **Lasso Regression ($\alpha=0.8$)** | $0.831$ | $0.824$ | $5,980$ | $7,520$ | Sparse Feature Drop |
| **Decision Tree Regressor** | $0.942$ | $0.789$ | $7,110$ | $9,340$ | High Overfitting |
| **Random Forest Regressor** | $0.938$ | **$0.914$** | **$3,840$** | **$5,120$** | **Optimal Champion** |
| **Gradient Boosting (GBM)** | $0.946$ | $0.908$ | $4,020$ | $5,310$ | Production Viable |

---

### <a id="importance"></a>🔬 FEATURE IMPORTANCE & VARIANCE ATTRIBUTION

```text
Feature Impact Distribution (Random Forest Gini Impurity Reduction)
══════════════════════════════════════════════════════════════════════════
Years of Experience   ████████████████████████████████████  [48.6%]
Job Level Tier        ██████████████████████               [29.4%]
Department Sector     ███████                               [9.8%]
Education Credential  █████                                 [7.1%]
Performance Rating    ███                                   [5.1%]
══════════════════════════════════════════════════════════════════════════
