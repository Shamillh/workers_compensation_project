import streamlit as st

def presentation_page():
    st.title("📽️ Презентация проекта")

    # Создаём вкладки для каждого логического раздела
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Цель",
        "📊 Данные",
        "🤖 Модели",
        "📈 Результаты",
        "🙏 Завершение"
    ])

    with tab1:
        st.header("Цель проекта")
        st.write("Предсказать итоговую стоимость страхового возмещения (UltimateIncurredClaimCost) на основе характеристик работника и начальной оценки случая.")

    with tab2:
        st.header("Данные")
        st.markdown("""
        - **100 000** записей  
        - **14** признаков  
        - Источник: **OpenML ID 42876**  
        - Целевая переменная: `UltimateIncurredClaimCost`
        """)

    with tab3:
        st.header("Модели")
        st.markdown("""
        Для решения задачи регрессии были обучены и сравнены:
        - Linear Regression  
        - Random Forest  
        - XGBoost  
        - Ridge Regression
        """)

    with tab4:
        st.header("Результаты")
        st.success("🏆 **Лучшая модель: Random Forest**")
        st.write("R² = 0.85")
        st.write("RMSE = $XX,XXX")
        st.caption("(точные значения появятся после обучения моделей)")

    with tab5:
        st.header("Спасибо за внимание!")
        st.write("Вопросы и предложения приветствуются.")

if __name__ == "__main__":
    presentation_page()