import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

st.set_page_config(page_title="Анализ и модель", layout="wide")


def main():
    st.title("📊 Прогнозирование страховых выплат")

    # Кнопка загрузки данных
    if st.button("Загрузить данные"):
        with st.spinner("Загрузка 100,000 записей..."):
            # ------------------- БЛОК С ОБРАБОТКОЙ ОШИБОК -------------------
            df = None

            csv_path = "Data/workers_compensation.csv"
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    st.success(f"✅ Данные загружены из локального файла: {csv_path}")
                except Exception as csv_err:
                    st.error(f"Ошибка при чтении CSV: {csv_err}")
                    st.stop()
            else:
                try:
                    data = fetch_openml(data_id=42876, as_frame=True, parser='auto')
                    df = data.frame
                    st.success("✅ Данные успешно загружены с OpenML!")
                except Exception as e:
                    st.error(f"Не удалось загрузить данные ни из CSV, ни с OpenML: {e}")
                    st.stop()

            if df is None:
                st.error("Не удалось загрузить данные.")
                st.stop()
            # ----------------------------------------------------------------

            # Простая предобработка - работа с датами
            df['DateTimeOfAccident'] = pd.to_datetime(df['DateTimeOfAccident'])
            df['DateReported'] = pd.to_datetime(df['DateReported'])
            df['ReportingDelay'] = (df['DateReported'] - df['DateTimeOfAccident']).dt.days
            df = df.drop(columns=['DateTimeOfAccident', 'DateReported'])

            # Кодирование категорий
            label_encoders = {}
            for col in ['Gender', 'MaritalStatus', 'PartTimeFullTime', 'ClaimDescription']:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
            st.session_state.label_encoders = label_encoders

            st.session_state.df = df
            st.success("✅ Данные загружены!")

            # Проверка на выбросы
            st.subheader("Проверка качества данных")
            col1, col2 = st.columns(2)

            with col1:
                negative_target = (df['UltimateIncurredClaimCost'] < 0).sum()
                if negative_target > 0:
                    st.warning(f"Найдено {negative_target} записей с отрицательной стоимостью")
                else:
                    st.success("Нет отрицательных значений в целевой переменной")

            with col2:
                missing_values = df.isnull().sum().sum()
                if missing_values > 0:
                    st.warning(f"Найдено {missing_values} пропущенных значений")
                else:
                    st.success("Нет пропущенных значений")

    # Если данные загружены
    if 'df' in st.session_state:
        df = st.session_state.df

        # Показываем данные
        st.subheader("Данные")
        st.write(df.head())

        if st.button("Обучить модель"):
            with st.spinner("Подготовка данных..."):
                X = df.drop('UltimateIncurredClaimCost', axis=1)
                y = df['UltimateIncurredClaimCost']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

                scaler = StandardScaler()
                X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
                X_test[num_cols] = scaler.transform(X_test[num_cols])

            # Определяем все модели (4 штуки)
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
                'Linear Regression': LinearRegression(),
                'Gradient Boosting (XGBoost)': XGBRegressor(n_estimators=50, random_state=42, verbosity=0),
                'Ridge Regression': Ridge(alpha=1.0, random_state=42)
            }

            progress_container = st.container()
            results = []
            best_model = None
            best_rmse = np.inf
            best_model_obj = None

            for name, model in models.items():
                with progress_container:
                    status_msg = st.info(f"⚙️ Обучаем модель: **{name}**...")
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)

                    results.append({
                        'Модель': name,
                        'MAE': f"${mae:,.0f}",
                        'RMSE': f"${rmse:,.0f}",
                        'R² Score': f"{r2:.4f}"
                    })

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = name
                        best_model_obj = model
                        st.session_state.model = model
                        st.session_state.X_columns = X.columns.tolist()
                        st.session_state.scaler = scaler
                        st.session_state.num_cols = num_cols

                    status_msg.success(f"✅ Модель **{name}** обучена (RMSE: ${rmse:,.0f})")

                except Exception as e:
                    status_msg.error(f"❌ Ошибка при обучении **{name}**: {e}")
                    results.append({
                        'Модель': name,
                        'MAE': "N/A",
                        'RMSE': "N/A",
                        'R² Score': "Ошибка"
                    })

            st.subheader("📊 Сравнение моделей")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            st.success(f"Лучшая модель: **{best_model}** с RMSE = ${best_rmse:,.0f}")

            # График сравнения RMSE
            fig, ax = plt.subplots(figsize=(10, 6))
            models_names = [r['Модель'] for r in results]
            rmse_values = [float(r['RMSE'].replace('$', '').replace(',', '')) for r in results]
            bars = ax.bar(models_names, rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_xlabel('Модели')
            ax.set_ylabel('RMSE ($)')
            ax.set_title('Сравнение RMSE моделей')
            ax.tick_params(axis='x', rotation=45)
            for bar, value in zip(bars, rmse_values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_height() * 0.01,
                        f'${value:,.0f}', ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("📈 Качество предсказаний лучшей модели")
            best_y_pred = best_model_obj.predict(X_test)
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.scatter(y_test, best_y_pred, alpha=0.3, s=1)
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax2.set_xlabel("Реальные значения ($)")
            ax2.set_ylabel("Предсказанные значения ($)")
            ax2.set_title(f"{best_model}: Предсказания vs Реальные значения")
            plt.tight_layout()
            st.pyplot(fig2)

            if best_model in ['Random Forest', 'Gradient Boosting (XGBoost)']:
                importance = pd.DataFrame({
                    'Признак': X.columns,
                    'Важность': best_model_obj.feature_importances_
                }).sort_values('Важность', ascending=False).head(10)
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.barh(importance['Признак'], importance['Важность'])
                ax3.set_xlabel("Важность")
                ax3.set_title(f"Топ-10 важных признаков ({best_model})")
                st.pyplot(fig3)

        # Форма для предсказания
        if 'model' in st.session_state:
            st.markdown("---")
            st.subheader("Предсказать новый случай")
            st.info(f"💡 Используется модель: **{st.session_state.model.__class__.__name__}**")

            if 'label_encoders' not in st.session_state:
                st.error("❌ Сначала загрузите данные и обучите модель")
                st.stop()

            enc = st.session_state.label_encoders

            with st.form("predict_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    age = st.number_input("Возраст", min_value=13, max_value=76, value=35)
                    gender_str = st.selectbox("Пол", list(enc['Gender'].classes_))
                    marital_str = st.selectbox("Семейное положение", list(enc['MaritalStatus'].classes_))
                    parttime_str = st.selectbox("Тип занятости", list(enc['PartTimeFullTime'].classes_))
                    claim_desc_str = st.selectbox("Описание заявки", list(enc['ClaimDescription'].classes_))
                with col2:
                    dependent_children = st.number_input("Детей на иждивении", 0, 10, 0)
                    dependents_other = st.number_input("Других иждивенцев", 0, 10, 0)
                    weekly_pay = st.number_input("Еженедельная зарплата ($)", 0, 5000, 500)
                with col3:
                    hours_week = st.number_input("Часов в неделю", 0, 80, 40)
                    days_week = st.number_input("Дней в неделю", 1, 7, 5)
                    initial_estimate = st.number_input("Начальная оценка ($)", 0, 100000, 5000)
                    accident_month = st.slider("Месяц происшествия", 1, 12, 6)
                    accident_day = st.slider("День недели (0=пн, 6=вс)", 0, 6, 2)
                    reporting_delay = st.number_input("Задержка отчетности (дней)", 0, 365, 7)

                submitted = st.form_submit_button("Предсказать стоимость", use_container_width=True)

                if submitted:
                    gender_enc = enc['Gender'].transform([gender_str])[0]
                    marital_enc = enc['MaritalStatus'].transform([marital_str])[0]
                    parttime_enc = enc['PartTimeFullTime'].transform([parttime_str])[0]
                    claim_enc = enc['ClaimDescription'].transform([claim_desc_str])[0]

                    feature_dict = {}
                    for col in st.session_state.X_columns:
                        if col == 'Age':
                            feature_dict[col] = age
                        elif col == 'Gender':
                            feature_dict[col] = gender_enc
                        elif col == 'MaritalStatus':
                            feature_dict[col] = marital_enc
                        elif col == 'DependentChildren':
                            feature_dict[col] = dependent_children
                        elif col == 'DependentsOther':
                            feature_dict[col] = dependents_other
                        elif col == 'WeeklyPay':
                            feature_dict[col] = weekly_pay
                        elif col == 'PartTimeFullTime':
                            feature_dict[col] = parttime_enc
                        elif col == 'HoursWorkedPerWeek':
                            feature_dict[col] = hours_week
                        elif col == 'DaysWorkedPerWeek':
                            feature_dict[col] = days_week
                        elif col == 'ClaimDescription':
                            feature_dict[col] = claim_enc
                        elif col == 'InitialCaseEstimate':
                            feature_dict[col] = initial_estimate
                        elif col == 'AccidentMonth':
                            feature_dict[col] = accident_month
                        elif col == 'AccidentDayOfWeek':
                            feature_dict[col] = accident_day
                        elif col == 'ReportingDelay':
                            feature_dict[col] = reporting_delay
                        else:
                            st.warning(f"Неизвестный признак: {col}, заменён на 0")
                            feature_dict[col] = 0

                    input_data = pd.DataFrame(
                        [[feature_dict[col] for col in st.session_state.X_columns]],
                        columns=st.session_state.X_columns
                    )
                    input_data[st.session_state.num_cols] = st.session_state.scaler.transform(
                        input_data[st.session_state.num_cols]
                    )
                    prediction = st.session_state.model.predict(input_data)[0]
                    prediction = max(0, prediction)
                    st.success(f"### Прогнозируемая стоимость возмещения: **${prediction:,.2f}**")
                    st.caption(f"Модель: {st.session_state.model.__class__.__name__} | "
                               f"Возраст: {age} | Зарплата: ${weekly_pay} | "
                               f"Начальная оценка: ${initial_estimate:,.0f}")


if __name__ == "__main__":
    main()