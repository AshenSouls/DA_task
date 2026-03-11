"""
Кредитный скоринг — Streamlit приложение
=========================================
Веб-интерфейс для оценки кредитного риска клиента на основе
обученной модели логистической регрессии с WoE-кодированием.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import math

# ─────────────────────────────────────────────────────────────────────────────
# Настройки страницы
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Кредитный скоринг",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
    }
    .risk-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .param-help {
        background-color: #e8f4fd;
        padding: 0.8rem;
        border-radius: 0.5rem;
        border-left: 3px solid #4C9BE8;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 50%, #F2994A 75%, #eb3349 100%);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка модели и артефактов
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Загружает модель и все артефакты."""
    model = joblib.load('model.pkl')
    woe_mapping = pd.read_csv('woe_mapping.csv')
    
    with open('bin_edges.json', 'r') as f:
        bin_edges = json.load(f)
    
    with open('feature_cols.json', 'r') as f:
        feature_cols = json.load(f)
    
    scorecard = pd.read_csv('scorecard.csv')
    
    return model, woe_mapping, bin_edges, feature_cols, scorecard


def get_bin_label(value: float, edges: list, feature: str) -> str:
    """Определяет бин для значения признака."""
    edges = [float(e) if e != float('inf') and e != float('-inf') else e for e in edges]
    
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        
        if left == float('-inf'):
            left = -math.inf
        if right == float('inf'):
            right = math.inf
            
        if left < value <= right:
            # Формируем строку бина в формате pandas Interval
            left_str = str(left) if left != -math.inf else '-inf'
            right_str = str(right) if right != math.inf else 'inf'
            return f"({left_str}, {right_str}]"
    
    # Если значение меньше или равно первой границе
    return f"(-inf, {edges[1]}]"


def get_woe_value(woe_mapping: pd.DataFrame, feature: str, bin_label: str) -> float:
    """Получает WoE значение для признака и бина."""
    import re
    
    mask = (woe_mapping['features'] == feature)
    subset = woe_mapping[mask]
    
    # Парсим границы нашего бина
    match = re.match(r'\(([^,]+),\s*([^\]]+)\]', bin_label)
    if match:
        left_str, right_str = match.groups()
        try:
            our_left = float(left_str) if left_str not in ['-inf', 'inf'] else (float('-inf') if left_str == '-inf' else float('inf'))
            our_right = float(right_str) if right_str not in ['-inf', 'inf'] else (float('-inf') if right_str == '-inf' else float('inf'))
        except:
            our_left, our_right = float('-inf'), float('inf')
    else:
        our_left, our_right = float('-inf'), float('inf')
    
    # Ищем бин в mapping, который покрывает наше значение (середину интервала)
    test_value = (our_left + our_right) / 2 if not (math.isinf(our_left) or math.isinf(our_right)) else (our_right if math.isinf(our_left) else our_left)
    
    for _, row in subset.iterrows():
        bin_str = str(row['bin'])
        bin_match = re.match(r'\(([^,]+),\s*([^\]]+)\]', bin_str)
        if bin_match:
            left_s, right_s = bin_match.groups()
            try:
                woe_left = float(left_s) if left_s not in ['-inf', 'inf'] else (float('-inf') if left_s == '-inf' else float('inf'))
                woe_right = float(right_s) if right_s not in ['-inf', 'inf'] else (float('-inf') if right_s == '-inf' else float('inf'))
                
                # Проверяем попадание test_value в интервал
                if woe_left < test_value <= woe_right:
                    return row['woe']
                # Или перекрытие интервалов
                if our_left < woe_right and woe_left < our_right:
                    return row['woe']
            except:
                continue
    
    # Резервный вариант — строковое совпадение
    for _, row in subset.iterrows():
        if bin_label in str(row['bin']) or str(row['bin']) in bin_label:
            return row['woe']
    
    # Если не найдено, возвращаем среднее WoE
    if len(subset) > 0:
        return subset['woe'].mean()
    return 0.0


def predict_default_probability(model, woe_mapping, bin_edges, feature_cols, input_data: dict) -> float:
    """Предсказывает вероятность дефолта."""
    woe_values = []
    
    for feature in feature_cols:
        value = input_data[feature]
        edges = bin_edges[feature]
        bin_label = get_bin_label(value, edges, feature)
        woe = get_woe_value(woe_mapping, feature, bin_label)
        woe_values.append(woe)
    
    # Формируем DataFrame с правильными названиями колонок
    woe_cols = [f"woe_bin_{f}" for f in feature_cols]
    X = pd.DataFrame([woe_values], columns=woe_cols)
    
    # Предсказание
    prob = model.predict_proba(X)[0][1]
    return prob


def calculate_credit_score(prob_default: float, base_score: int = 650, pdo: float = 72.13) -> int:
    """Рассчитывает кредитный скор на основе вероятности дефолта."""
    # Избегаем деления на ноль и log(0)
    prob_default = max(0.001, min(0.999, prob_default))
    odds = (1 - prob_default) / prob_default
    score = base_score + pdo * np.log(odds)
    # Ограничиваем скор в разумных пределах (300-850)
    score = max(300, min(850, score))
    return int(round(score))


def get_risk_level(prob_default: float) -> tuple:
    """Определяет уровень риска и цвет."""
    if prob_default < 0.1:
        return "🟢 Низкий", "risk-low", "Одобрить"
    elif prob_default < 0.3:
        return "🟡 Средний", "risk-medium", "Требуется дополнительная проверка"
    else:
        return "🔴 Высокий", "risk-high", "Отказать"


def calculate_max_loan(monthly_income: float, current_payments: float, 
                       annual_rate: float = 0.18, loan_term_months: int = 60) -> dict:
    """
    Рассчитывает максимальную сумму кредита на основе дохода.
    
    Банки обычно одобряют кредит, если DTI (долговая нагрузка) <= 50%.
    Консервативная оценка: DTI <= 40%.
    """
    if monthly_income <= 0:
        return {"max_payment": 0, "max_loan": 0, "conservative_loan": 0}
    
    # Максимальный допустимый платёж (50% от дохода минус текущие платежи)
    max_dti = 0.50
    conservative_dti = 0.40
    
    max_available_payment = monthly_income * max_dti - current_payments
    conservative_payment = monthly_income * conservative_dti - current_payments
    
    # Если уже превышен лимит
    if max_available_payment <= 0:
        return {"max_payment": 0, "max_loan": 0, "conservative_loan": 0}
    
    # Расчёт суммы кредита по аннуитетной формуле
    # P = A * [(1 - (1 + r)^(-n)) / r]
    # где P - сумма кредита, A - ежемесячный платёж, r - месячная ставка, n - срок в месяцах
    monthly_rate = annual_rate / 12
    
    def payment_to_loan(payment, rate, months):
        if rate == 0:
            return payment * months
        return payment * ((1 - (1 + rate) ** (-months)) / rate)
    
    max_loan = payment_to_loan(max_available_payment, monthly_rate, loan_term_months)
    conservative_loan = payment_to_loan(max(0, conservative_payment), monthly_rate, loan_term_months)
    
    return {
        "max_payment": max(0, max_available_payment),
        "max_loan": max(0, max_loan),
        "conservative_loan": max(0, conservative_loan)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Основной интерфейс
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown('<h1 class="main-header">💳 Кредитный скоринг</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Узнайте ваш кредитный рейтинг за 1 минуту</p>', unsafe_allow_html=True)
    
    # Сайдбар с информацией
    with st.sidebar:
        st.markdown("### ℹ️ Как это работает?")
        st.markdown("""
        1. **Заполните форму** — укажите ваши реальные данные
        2. **Нажмите кнопку** — система проанализирует информацию
        3. **Получите результат** — кредитный скор и рекомендации
        
        ---
        
        ### 📊 Что означает скор?
        
        | Скор | Оценка |
        |------|--------|
        | 750+ | Отличный |
        | 700-749 | Хороший |
        | 650-699 | Средний |
        | 600-649 | Низкий |
        | <600 | Плохой |
        
        ---
        
        ### 🔒 Конфиденциальность
        
        Ваши данные **не сохраняются** и используются 
        только для расчёта прямо сейчас.
        """)
    
    # Проверяем наличие файлов модели
    try:
        model, woe_mapping, bin_edges, feature_cols, scorecard = load_model()
    except FileNotFoundError as e:
        st.error("""
        ⚠️ **Файлы модели не найдены!**
        
        Пожалуйста, выполните ячейку сохранения модели в Jupyter Notebook:
        - `model.pkl`
        - `woe_mapping.csv`
        - `bin_edges.json`
        - `feature_cols.json`
        - `scorecard.csv`
        """)
        st.stop()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Форма ввода данных
    # ─────────────────────────────────────────────────────────────────────────
    
    st.markdown("### 📋 Данные клиента")
    st.caption("Введите ваши данные — система автоматически рассчитает кредитный рейтинг")
    
    with st.form("scoring_form"):
        
        # ═══════════════════════════════════════════════════════════════════
        # БЛОК 1: Персональные данные
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("#### 👤 Персональные данные")
        
        age = st.slider(
            "🎂 Ваш возраст",
            min_value=18,
            max_value=88,
            value=35
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # БЛОК 2: Доходы
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("#### 💰 Ваши ежемесячные доходы")
        st.caption("Укажите все источники дохода (до вычета налогов)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            salary = st.number_input(
                "💼 Зарплата",
                min_value=0,
                max_value=10_000_000,
                value=50000,
                step=5000,
                help="Официальная зарплата на основном месте работы"
            )
        
        with col2:
            other_income = st.number_input(
                "📈 Дополнительный доход",
                min_value=0,
                max_value=10_000_000,
                value=0,
                step=1000,
                help="Подработка, пенсия, алименты, аренда, пособия и т.д."
            )
        
        # Показываем итоговый доход
        total_income = salary + other_income
        if total_income > 0:
            st.info(f"📊 **Общий доход:** {total_income:,}₸ в месяц".replace(",", " "))
        
        # ═══════════════════════════════════════════════════════════════════
        # БЛОК 3: Кредитная нагрузка
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("#### 💳 Ваши кредиты и карты")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ежемесячные платежи по кредитам**")
            st.caption("Сумма всех платежей: ипотека, автокредит, потребительские кредиты")
            
            monthly_payments = st.number_input(
                "Платежи по кредитам (₸/мес)",
                min_value=0,
                max_value=10_000_000,
                value=0,
                step=1000,
                label_visibility="collapsed"
            )
            
            # Автоматический расчёт DTI
            if total_income > 0:
                debt_ratio = monthly_payments / total_income
                if debt_ratio < 0.3:
                    st.success(f"✅ Нагрузка: {debt_ratio*100:.0f}% от дохода — это хорошо!")
                elif debt_ratio < 0.5:
                    st.warning(f"⚠️ Нагрузка: {debt_ratio*100:.0f}% от дохода — умеренно")
                else:
                    st.error(f"❌ Нагрузка: {debt_ratio*100:.0f}% от дохода — очень высокая!")
            else:
                debt_ratio = 0.0
        
        with col2:
            st.markdown("**Кредитные карты**")
            st.caption("Текущий долг и общий лимит по всем картам")
            
            card_debt = st.number_input(
                "Текущий долг по картам (₸)",
                min_value=0,
                max_value=10_000_000,
                value=0,
                step=1000
            )
            
            card_limit = st.number_input(
                "Общий лимит по картам (₸)",
                min_value=0,
                max_value=10_000_000,
                value=100000,
                step=10000,
                help="Сумма лимитов всех ваших кредитных карт"
            )
            
            # Автоматический расчёт utilization
            if card_limit > 0:
                revolving_utilization = card_debt / card_limit
                if revolving_utilization <= 0.3:
                    st.success(f"✅ Использовано {revolving_utilization*100:.0f}% лимита — отлично!")
                elif revolving_utilization <= 0.7:
                    st.warning(f"⚠️ Использовано {revolving_utilization*100:.0f}% лимита")
                elif revolving_utilization <= 1.0:
                    st.error(f"❌ Использовано {revolving_utilization*100:.0f}% лимита — много!")
                else:
                    st.error(f"🚨 Превышение лимита! ({revolving_utilization*100:.0f}%)")
            else:
                revolving_utilization = 0.0
                st.info("💡 Нет кредитных карт")
        
        # ═══════════════════════════════════════════════════════════════════
        # БЛОК 4: Кредитная история
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("#### 📅 Кредитная история (за последние 2 года)")
        st.caption("Были ли у вас просрочки по платежам?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            times_30_59_late = st.selectbox(
                "Просрочки 30-59 дней",
                options=list(range(0, 11)),
                index=0,
                help="Небольшие задержки платежей"
            )
        
        with col2:
            times_60_89_late = st.selectbox(
                "Просрочки 60-89 дней",
                options=list(range(0, 11)),
                index=0,
                help="Существенные задержки"
            )
        
        with col3:
            times_90_late = st.selectbox(
                "Просрочки 90+ дней",
                options=list(range(0, 11)),
                index=0,
                help="Серьёзные просрочки — сильно влияют на рейтинг"
            )
        
        # Предупреждение о просрочках
        total_late = times_30_59_late + times_60_89_late + times_90_late
        if total_late == 0:
            st.success("✅ Отличная кредитная история — просрочек нет!")
        elif total_late <= 2:
            st.warning(f"⚠️ Найдено {total_late} просрочек — это снизит рейтинг")
        else:
            st.error(f"❌ Найдено {total_late} просрочек — серьёзно влияет на рейтинг")
        
        st.markdown("---")
        
        submitted = st.form_submit_button(
            "🔍 Рассчитать кредитный рейтинг",
            use_container_width=True,
            type="primary"
        )
        
        # Сохраняем итоговый доход для модели
        monthly_income = total_income
    
    # ─────────────────────────────────────────────────────────────────────────
    # Результаты
    # ─────────────────────────────────────────────────────────────────────────
    
    if submitted:
        # Собираем входные данные (7 признаков)
        input_data = {
            'RevolvingUtilizationOfUnsecuredLines': revolving_utilization,
            'NumberOfTime30-59DaysPastDueNotWorse': times_30_59_late,
            'age': age,
            'NumberOfTimes90DaysLate': times_90_late,
            'NumberOfTime60-89DaysPastDueNotWorse': times_60_89_late,
            'MonthlyIncome': monthly_income,
            'DebtRatio': debt_ratio,
        }
        
        # Вычисляем вероятность дефолта
        with st.spinner("Анализируем данные..."):
            prob_default = predict_default_probability(
                model, woe_mapping, bin_edges, feature_cols, input_data
            )
            credit_score = calculate_credit_score(prob_default)
            risk_level, risk_class, decision = get_risk_level(prob_default)
        
        st.markdown("---")
        st.markdown("### 📊 Результаты оценки")
        
        # Три колонки с метриками
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.9;">Кредитный скор</div>
                <div style="font-size: 2.5rem; font-weight: 700;">{credit_score}</div>
                <div style="font-size: 0.8rem; opacity: 0.8;">из 850</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card {risk_class}">
                <div style="font-size: 0.9rem; opacity: 0.9;">Уровень риска</div>
                <div style="font-size: 1.8rem; font-weight: 700;">{risk_level}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            decision_emoji = "✅" if "Одобрить" in decision else ("⚠️" if "проверка" in decision else "❌")
            st.markdown(f"""
            <div class="metric-card {'risk-low' if 'Одобрить' in decision else ('risk-medium' if 'проверка' in decision else 'risk-high')}">
                <div style="font-size: 0.9rem; opacity: 0.9;">Решение</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{decision_emoji} {decision}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Вероятность дефолта с прогресс-баром
        st.markdown("#### 📈 Вероятность дефолта")
        st.progress(min(prob_default, 1.0))
        st.markdown(f"**{prob_default * 100:.1f}%** — вероятность невозврата кредита в течение 2 лет")
        
        # ═══════════════════════════════════════════════════════════════════
        # Расчёт рекомендуемой суммы кредита
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("#### 💰 Какой кредит вы можете взять?")
        st.caption("Ставка 20% годовых")
        
        # Функция для расчёта платежа по кредиту
        def loan_to_payment(loan, rate, months):
            if rate == 0 or loan <= 0:
                return loan / months if months > 0 else 0
            return loan * (rate * (1 + rate) ** months) / ((1 + rate) ** months - 1)
        
        monthly_rate = 0.20 / 12
        
        # Заранее рассчитываем все варианты
        loan_variants = []
        for term_name, term_months in [("6 мес", 6), ("12 мес", 12), ("24 мес", 24)]:
            info = calculate_max_loan(
                monthly_income=monthly_income,
                current_payments=monthly_payments,
                annual_rate=0.20,
                loan_term_months=term_months
            )
            cons_pmt = loan_to_payment(info['conservative_loan'], monthly_rate, term_months)
            max_pmt = loan_to_payment(info['max_loan'], monthly_rate, term_months)
            loan_variants.append({
                "term": term_name,
                "months": term_months,
                "conservative_loan": info['conservative_loan'],
                "conservative_payment": cons_pmt,
                "max_loan": info['max_loan'],
                "max_payment": max_pmt,
                "available": info['max_loan'] > 0
            })
        
        # Проверяем доступность кредита
        if any(v["available"] for v in loan_variants):
            # Таблица с комфортными суммами
            st.markdown("##### 🟢 Комфортный кредит (нагрузка до 40%)")
            
            cols = st.columns(3)
            for i, v in enumerate(loan_variants):
                with cols[i]:
                    if v["conservative_loan"] > 0:
                        st.markdown(f"""
                        <div class="metric-card risk-low" style="text-align: center; padding: 15px;">
                            <div style="font-size: 1rem; font-weight: 600;">{v['term']}</div>
                            <div style="font-size: 1.5rem; font-weight: 700; margin: 8px 0;">{v['conservative_loan']:,.0f}₸</div>
                            <div style="font-size: 0.85rem;">Платёж: {v['conservative_payment']:,.0f}₸/мес</div>
                        </div>
                        """.replace(",", " "), unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center; padding: 15px; opacity: 0.5;">
                            <div style="font-size: 1rem;">{v['term']}</div>
                            <div style="font-size: 1.2rem; margin: 8px 0;">Недоступно</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Таблица с максимальными суммами
            st.markdown("##### 🟡 Максимальный кредит (нагрузка до 50%)")
            
            cols = st.columns(3)
            for i, v in enumerate(loan_variants):
                with cols[i]:
                    if v["max_loan"] > 0:
                        st.markdown(f"""
                        <div class="metric-card risk-medium" style="text-align: center; padding: 15px;">
                            <div style="font-size: 1rem; font-weight: 600;">{v['term']}</div>
                            <div style="font-size: 1.5rem; font-weight: 700; margin: 8px 0;">{v['max_loan']:,.0f}₸</div>
                            <div style="font-size: 0.85rem;">Платёж: {v['max_payment']:,.0f}₸/мес</div>
                        </div>
                        """.replace(",", " "), unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center; padding: 15px; opacity: 0.5;">
                            <div style="font-size: 1rem;">{v['term']}</div>
                            <div style="font-size: 1.2rem; margin: 8px 0;">Недоступно</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Пояснение
            available_payment = monthly_income * 0.50 - monthly_payments
            st.info(f"""
            📊 **Как это работает:**  
            - Ваш доход: **{monthly_income:,}₸/мес**  
            - Текущие платежи по кредитам: **{monthly_payments:,}₸/мес** ({debt_ratio*100:.0f}% от дохода)  
            - Свободно для нового кредита: **{max(0, available_payment):,.0f}₸/мес** (до 50% нагрузки)
            """.replace(",", " "))
            
            # Предупреждение при высоком DTI
            if debt_ratio > 0.4:
                st.warning(f"""
                ⚠️ **Внимание:** Ваша текущая долговая нагрузка уже {debt_ratio*100:.0f}%.  
                Банки могут отказать или предложить меньшую сумму.
                """)
        else:
            st.error("""
            ❌ **Кредит недоступен**
            
            Ваша текущая долговая нагрузка превышает 50% от дохода.  
            Рекомендуется сначала погасить часть существующих кредитов.
            """)
        
        # Интерпретация
        st.markdown("---")
        st.markdown("#### 💡 Интерпретация")
        
        if prob_default < 0.1:
            st.success("""
            **Отличный кандидат для кредитования!**
            
            Клиент демонстрирует низкий уровень риска. Рекомендуется одобрить кредит 
            на стандартных условиях.
            """)
        elif prob_default < 0.3:
            st.warning("""
            **Требуется дополнительный анализ.**
            
            Клиент находится в зоне умеренного риска. Рекомендуется:
            - Запросить дополнительные документы
            - Рассмотреть возможность поручительства
            - Предложить сниженный лимит кредитования
            """)
        else:
            st.error("""
            **Высокий риск невозврата.**
            
            Клиент демонстрирует высокий уровень риска. Рекомендуется:
            - Отказать в кредите **или**
            - Предложить альтернативный продукт с залогом
            - Значительно снизить сумму кредита
            """)
        
        # Детали расчёта (expander)
        with st.expander("🔬 Детали расчёта"):
            st.markdown("**Ваши данные:**")
            
            details_df = pd.DataFrame({
                'Параметр': [
                    'Возраст',
                    'Зарплата',
                    'Дополнительный доход',
                    'Общий доход',
                    'Платежи по кредитам',
                    'Долговая нагрузка',
                    'Долг по картам',
                    'Лимит по картам',
                    'Использование лимита',
                    'Просрочки (30-59 дней)',
                    'Просрочки (60-89 дней)',
                    'Просрочки (90+ дней)'
                ],
                'Значение': [
                    f"{age} лет",
                    f"{salary:,}₸".replace(",", " "),
                    f"{other_income:,}₸".replace(",", " "),
                    f"{monthly_income:,}₸".replace(",", " "),
                    f"{monthly_payments:,}₸".replace(",", " "),
                    f"{debt_ratio * 100:.1f}%",
                    f"{card_debt:,}₸".replace(",", " "),
                    f"{card_limit:,}₸".replace(",", " "),
                    f"{revolving_utilization * 100:.1f}%",
                    times_30_59_late,
                    times_60_89_late,
                    times_90_late
                ]
            })
            st.table(details_df)
            
            st.markdown(f"""
            **Формула кредитного скора:**
            
            $Score = 650 + 72.13 \\times \\ln\\left(\\frac{{1 - P_{{default}}}}{{P_{{default}}}}\\right)$
            
            где $P_{{default}} = {prob_default:.4f}$
            """)


if __name__ == "__main__":
    main()
