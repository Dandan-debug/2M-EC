import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import shap
import matplotlib.pyplot as plt

# 显示图片（图片在上，标题在下）
st.markdown("""
    <img src="https://github.com/Dandan-debug/2M-EC/raw/main/endometrial.svg" width="100" alt="Endometrial Cancer Model Image" style="display: block; margin: 0 auto 20px;">
    <h1 style="font-weight: bold; font-size: 50px; text-align: center; margin: 0;">
        2M-EC Predictive Platform
    </h1>
""", unsafe_allow_html=True)

# 在标题的 st.markdown 之后，描述文字的 st.markdown 之前，插入：
st.info("🆕 **The updated v1.0.1 website is now available at** https://2e-mc-web-1-0-1.streamlit.app/ — you can now upload MS data Excel files directly on the platform.")

# 显示描述文本
st.markdown("""
    <p style='text-align: left; font-size: 16px; margin-bottom: 28px;'>
        The 2M-EC (Multimodal Multilevel Endometrial Cancer) is a patient-first platform for the early screening of endometrial cancer (EC).<br><br>
        Employs patient-centered models to calculate specific risk probabilities:<br>
        • <b>"CP"</b> (EC minimally invasive screening): Integrating cervicovaginal metabolic profiling and plasma molecular profiling with routine clinical indicators for early-stage cancer screening<br>
        • <b>"UCP"</b> (EC precision screening): Integrating uterine metabolic profiling, cervicovaginal metabolic profiling and plasma molecular profiling with routine clinical indicators for precision cancer screening<br><br>
        Input data includes:<br>
        • Patient baseline: demographics, medical history, ultrasound examination results, and plasma tumor markers (HE4 and CA125)<br>
        • Biofluid molecular omics: cervicovaginal metabolic omics, uterine metabolic omics, and plasma molecular omics<br><br>
        Risk screening results annotation:<br>
        • High-risk probability = Highest cancer probability across models<br>
        • Low-risk probability = 1 - Highest cancer probability<br><br>
        Select the model that best fits your requirements and submit: "CP" or "UCP".
    </p>
""", unsafe_allow_html=True)

# 加载标准器和模型
scalers = {
    'C': joblib.load('scaler_standard_C.pkl'),
    'P': joblib.load('scaler_standard_P.pkl'),
    'U': joblib.load('scaler_standard_U.pkl')
}

models = {
    'C': joblib.load('xgboost_C.pkl'),
    'P': joblib.load('xgboost_P.pkl'),
    'U': joblib.load('xgboost_U.pkl')
}

# 定义特征名称
display_features_to_scale = [
    'Age (years)',                                  # Age (e.g., 52 years)
    'Endometrial thickness (mm)',                   # Endometrial thickness in mm
    'HE4 (pmol/L)',                                 # HE4 level in pmol/L
    'Menopause (1=yes)',                            # Menopause status (1=yes)
    'HRT (Hormone Replacement Therapy, 1=yes)',     # HRT status (1=yes)
    'Endometrial heterogeneity (1=yes)',            # Endometrial heterogeneity (1=yes)
    'Uterine cavity occupation (1=yes)',            # Uterine cavity occupation (1=yes)
    'Uterine cavity occupying lesion with rich blood flow (1=yes)', # Uterine cavity occupying lesion with rich blood flow (1=yes)
    'Uterine cavity fluid (1=yes)'                  # Uterine cavity fluid (1=yes)
]

# 原始特征名称，用于标准化器
original_features_to_scale = [
    'CI_age', 'CI_endometrial thickness', 'CI_HE4', 'CI_menopause',
    'CI_HRT', 'CI_endometrial heterogeneity',
    'CI_uterine cavity occupation',
    'CI_uterine cavity occupying lesion with rich blood flow',
    'CI_uterine cavity fluid'
]

# 额外特征名称映射（移除 .0 后缀）
additional_features = {
    'C': ['CM4160.0','CM727.0','CM889.0','CM7441.0','CM995.0','CM7440.0','CM7439.0','CM734.0',
          'CM1857.0','CM6407.0','CM2920.0','CM729.0','CM628.0'],

    'P': ['PM816.0','PM846.0','PM120.0','PP408.0','PM883.0','PM801.0','PM578.0',
          'PP48.0','PM504.0','PP317.0','PM722.0','PM86.0','PP63.0','PP405.0',
          'PM574.0','PP434.0','PM163.0','PP81.0','PM461.0','PM571.0','PM88.0','PP378.0',
          'PM867.0','PP286.0','PM409.0','PP497.0','PM900.0','PM836.0','PP393.0',
          'PP653.0','PP456.0','PP75.0','PP488.0','PM887.0','PP640.0','PP344.0',
          'PM584.0','PM396.0','PM681.0','PP332.0','PM328.0','PM882.0','PM548.0',
          'PM832.0','PM232.0','PM285.0','PM104.0','PM379.0','PM782.0'],

    'U': ['UM7578.0', 'UM510.0', 'UM507.0', 'UM670.0', 'UM351.0',
          'UM5905.0', 'UM346.0', 'UM355.0', 'UM8899.0', 'UM1152.0',
          'UM5269.0', 'UM6437.0', 'UM5906.0', 'UM7622.0', 'UM8898.0',
          'UM2132.0', 'UM3513.0', 'UM790.0', 'UM8349.0', 'UM2093.0',
          'UM4210.0', 'UM3935.0', 'UM4256.0']
}

# 模型选择
selected_models = st.multiselect(
    "Select the model(s) to be used (you can select one or more)",
    options=['U', 'C', 'P'],
    default=['U']
)

# 获取用户输入
user_input = {}

# 定义特征输入
for i, feature in enumerate(display_features_to_scale):
    if "1=yes" in feature:  # 对于分类变量，限制输入为0或1
        user_input[original_features_to_scale[i]] = st.selectbox(f"{feature}:", options=[0, 1])
    else:  # 对于连续变量，使用数值输入框
        user_input[original_features_to_scale[i]] = st.number_input(f"{feature}:", min_value=0.0, value=0.0)

# 为每个选定的模型定义额外特征
for model_key in selected_models:
    for feature in additional_features[model_key]:
        # 允许保留较多小数位的输入
        user_input[feature] = st.number_input(f"{feature} ({model_key}):", min_value=0.0, format="%.9f")


# 预测按钮
if st.button("Submit"):
    # 定义模型预测结果存储字典
    model_predictions = {}
    shap_explanations = {}

    # 对选定的每个模型进行标准化和预测
    for model_key in selected_models:
        # 针对每个模型构建专用的输入数据
        model_input_df = pd.DataFrame([user_input])
        
        # 获取模型所需的特征列
        model_features = original_features_to_scale + additional_features[model_key]
        
        # 仅保留当前模型需要的特征
        model_input_df = model_input_df[model_features]
        
        # 对需要标准化的特征进行标准化
        scaled_features_df = pd.DataFrame(
            scalers[model_key].transform(model_input_df[original_features_to_scale]),
            columns=original_features_to_scale,
            index=model_input_df.index
        )
        
        # 将标准化后的特征与未标准化的额外特征合并
        final_input_df = pd.concat([scaled_features_df, model_input_df[additional_features[model_key]]], axis=1)

        # 使用模型进行预测
        predicted_proba = models[model_key].predict_proba(final_input_df)[0]
        predicted_class = models[model_key].predict(final_input_df)[0]
        
        # 保存预测结果
        model_predictions[model_key] = {
            'proba': predicted_proba,
            'class': predicted_class
        }

        # 计算 SHAP 值
        explainer = shap.TreeExplainer(models[model_key])
        shap_values_Explanation = explainer(final_input_df)  # 计算SHAP值
        # 保存 SHAP 解释对象
        shap_explanations[model_key] = shap_values_Explanation

        # 仅在符合条件时显示 SHAP 水平图
        if set(selected_models) == {'C', 'P'} or len(selected_models) == 3:
            # 绘制 SHAP 图
            st.subheader(f"SHAP Waterfall Plot for Model {model_key}")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values_Explanation[0], show=False)  
            st.pyplot(fig)
            plt.close(fig)

    # 处理其他的预测逻辑 (ENDOM screening 或 diagnosis)
    if len(selected_models) == 1:
        st.error("Error: Please select at least two models for CP/UCP screening or three models for ENDOM diagnosis.")

    elif len(selected_models) == 2 and set(selected_models) != {'C', 'P'}:
        st.error("Error: For ENDOM screening, please select both 'C' and 'P' models.")

    elif len(selected_models) == 2 and set(selected_models) == {'C', 'P'}:
        has_positive = any(model_predictions[model_key]['class'] == 1 for model_key in selected_models)
        max_proba = max(model_predictions[model_key]['proba'][1] for model_key in selected_models)
        if has_positive:
            st.write(f"ENDOM screening：{max_proba * 100:.2f}%- high risk")
        else:
            st.write(f"ENDOM screening：{max_proba * 100:.2f}%- low risk")

    elif len(selected_models) == 3:
        positive_count = sum(model_predictions[model_key]['class'] == 1 for model_key in selected_models)
        max_proba = max(model_predictions[model_key]['proba'][1] for model_key in selected_models)
        if positive_count >= 2:
            st.write(f"ENDOM diagnosis：{max_proba * 100:.2f}%- high risk")
        else:
            low_risk_proba = (1 - max_proba) * 100
            st.write(f"ENDOM diagnosis：{low_risk_proba:.2f}%- low risk")

    else:
        st.error("Error: Invalid number of models selected. Please select 2 models (C and P) for screening or 3 models for diagnosis.")
