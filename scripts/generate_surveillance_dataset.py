"""
Generate synthetic public health surveillance dataset for training nanochat.

This script creates a comprehensive dataset covering:
- Disease outbreak detection and monitoring
- Epidemiological trend analysis
- Public health risk assessment
- Surveillance report generation
- Health alert interpretation
- Vaccination coverage analysis
- Disease prevalence queries
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta

# Seed for reproducibility
random.seed(42)

# -----------------------------------------------------------------------------
# Disease and Health Condition Data
# -----------------------------------------------------------------------------

DISEASES = [
    "influenza", "COVID-19", "measles", "tuberculosis", "malaria",
    "dengue fever", "cholera", "hepatitis A", "hepatitis B", "HIV/AIDS",
    "pneumonia", "whooping cough", "mumps", "rubella", "chickenpox",
    "salmonellosis", "listeriosis", "E. coli infection", "norovirus",
    "West Nile virus", "Zika virus", "Lyme disease", "rabies",
    "meningitis", "pertussis", "typhoid fever", "yellow fever"
]

REGIONS = [
    "Northeast", "Southeast", "Midwest", "Southwest", "West Coast",
    "Pacific Northwest", "Great Lakes", "Mid-Atlantic", "New England",
    "Gulf Coast", "Mountain West", "Central Plains"
]

DEMOGRAPHICS = ["children under 5", "adolescents 12-17", "adults 18-64", "elderly 65+",
                "pregnant women", "healthcare workers", "immunocompromised individuals"]

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

SEASONS = ["winter", "spring", "summer", "fall"]

# -----------------------------------------------------------------------------
# Template Functions for Different Surveillance Tasks
# -----------------------------------------------------------------------------

def generate_outbreak_detection():
    """Generate questions about detecting disease outbreaks"""
    disease = random.choice(DISEASES)
    region = random.choice(REGIONS)
    cases = random.randint(50, 500)
    baseline = random.randint(10, 50)

    templates = [
        {
            "question": f"There are {cases} reported cases of {disease} in the {region} region over the past two weeks, compared to a baseline of {baseline} cases. Is this an outbreak?",
            "answer": f"Yes, this appears to be an outbreak. The current number of {cases} cases significantly exceeds the baseline of {baseline} cases (a {round((cases-baseline)/baseline*100)}% increase). This meets the epidemiological threshold for an outbreak, defined as cases exceeding expected levels for that time and place. Immediate investigation is warranted to:\n\n1. Confirm cases through laboratory testing\n2. Identify the source of infection\n3. Implement control measures\n4. Conduct contact tracing\n5. Assess risk to the broader population\n\nPublic health authorities should be notified immediately to coordinate response efforts."
        },
        {
            "question": f"What surveillance indicators suggest a {disease} outbreak in {region}?",
            "answer": f"Key surveillance indicators for a {disease} outbreak in {region} include:\n\n**Case-based indicators:**\n- Sudden increase in case counts above historical baseline\n- Clustering of cases in time and location\n- Unusual severity or mortality rates\n- Cases in unexpected age groups or demographics\n\n**Laboratory indicators:**\n- Increased positive test results\n- Detection of unusual strains or variants\n- Higher test positivity rates\n\n**Syndromic surveillance:**\n- Increases in emergency department visits for related symptoms\n- Spikes in over-the-counter medication sales\n- School or workplace absenteeism patterns\n\n**Environmental factors:**\n- Changes in vector populations (if vector-borne)\n- Contamination of water or food sources\n- Environmental conditions favoring transmission\n\nMonitoring these indicators together provides early warning of potential outbreaks."
        }
    ]

    return random.choice(templates)

def generate_trend_analysis():
    """Generate questions about analyzing epidemiological trends"""
    disease = random.choice(DISEASES)
    season = random.choice(SEASONS)
    demographic = random.choice(DEMOGRAPHICS)
    percentage = random.randint(15, 75)

    templates = [
        {
            "question": f"Analyze the trend: {disease} cases have increased by {percentage}% among {demographic} during {season}. What does this indicate?",
            "answer": f"This {percentage}% increase in {disease} cases among {demographic} during {season} indicates several important surveillance findings:\n\n**Epidemiological significance:**\n- The increase suggests enhanced transmission in this specific population\n- Seasonal patterns may be contributing to spread\n- This demographic may have specific risk factors or exposures\n\n**Public health implications:**\n1. **Targeted interventions needed:** Focus prevention efforts on {demographic}\n2. **Risk communication:** Develop age-appropriate messaging for this group\n3. **Healthcare preparedness:** Ensure clinical facilities are prepared for this demographic\n4. **Investigation priorities:** Identify specific transmission routes affecting this population\n\n**Recommended actions:**\n- Enhanced surveillance in this demographic group\n- Review vaccination coverage if vaccine-preventable\n- Assess social and environmental determinants\n- Compare with historical seasonal patterns\n- Implement targeted prevention strategies\n\nContinued monitoring is essential to determine if this represents a sustained trend or temporary fluctuation."
        },
        {
            "question": f"What factors might explain the seasonal increase in {disease} cases?",
            "answer": f"Multiple factors can explain seasonal increases in {disease} cases:\n\n**Environmental factors:**\n- Temperature and humidity changes affecting pathogen survival\n- Seasonal weather patterns influencing transmission\n- Vector population changes (for vector-borne diseases)\n- Indoor crowding during certain seasons\n\n**Behavioral factors:**\n- Seasonal gathering patterns (holidays, events)\n- Changes in travel and mobility\n- Indoor vs. outdoor activities\n- School schedules and attendance patterns\n\n**Host factors:**\n- Seasonal variation in immune function\n- Vitamin D levels (for winter increases)\n- Stress and lifestyle changes\n- Crowding in enclosed spaces\n\n**Pathogen factors:**\n- Seasonal mutations or strain changes\n- Survival advantages in certain conditions\n- Competition with other pathogens\n\n**Healthcare factors:**\n- Seasonal healthcare seeking behavior\n- Testing and surveillance intensity changes\n- Healthcare system capacity variations\n\nUnderstanding these factors helps predict future trends and implement timely interventions."
        }
    ]

    return random.choice(templates)

def generate_risk_assessment():
    """Generate questions about public health risk assessment"""
    disease = random.choice(DISEASES)
    region = random.choice(REGIONS)
    population = random.randint(100000, 1000000)

    templates = [
        {
            "question": f"Assess the public health risk of {disease} in a {region} community of {population:,} people with 5% vaccination coverage.",
            "answer": f"**Risk Assessment for {disease} in {region}**\n\n**Overall Risk Level: HIGH**\n\n**Population at Risk:**\n- Total population: {population:,}\n- Vaccinated (5%): {int(population * 0.05):,}\n- Unvaccinated/susceptible (95%): {int(population * 0.95):,}\n\n**Risk Factors:**\n1. **Low vaccination coverage (5%)** - Far below herd immunity threshold\n2. **Large susceptible population** - {int(population * 0.95):,} people at risk\n3. **High transmission potential** in this region\n\n**Potential Impact:**\n- **High attack rate expected** due to low immunity\n- **Healthcare system strain** from potential surge in cases\n- **Vulnerable populations** at severe disease risk\n- **Community transmission** likely to be sustained\n\n**Immediate Recommendations:**\n1. **Urgent vaccination campaign** targeting vulnerable groups\n2. **Enhanced surveillance** for early case detection\n3. **Healthcare preparedness** for potential surge\n4. **Risk communication** to increase vaccination uptake\n5. **Contact tracing capacity** establishment\n6. **Outbreak response plan** activation\n\n**Timeline:** Immediate action required within 2-4 weeks to prevent widespread outbreak.\n\n**Monitoring:** Daily case surveillance and weekly vaccination coverage updates essential."
        },
        {
            "question": f"What data do you need to conduct a comprehensive risk assessment for {disease}?",
            "answer": f"A comprehensive risk assessment for {disease} requires:\n\n**Epidemiological Data:**\n- Current case counts and trends\n- Historical baseline incidence rates\n- Attack rates by age, location, and demographics\n- Severity and case fatality rates\n- Transmission patterns and reproduction number (R₀)\n\n**Population Data:**\n- Total population size and density\n- Age distribution\n- Demographic characteristics\n- Vulnerable population counts\n- Population mobility patterns\n\n**Immunological Data:**\n- Vaccination coverage rates\n- Natural immunity levels (prior exposure)\n- Immunocompromised population size\n- Age-specific immunity profiles\n\n**Healthcare Data:**\n- Healthcare system capacity (beds, ICU, ventilators)\n- Healthcare workforce availability\n- Diagnostic testing capacity\n- Treatment availability and effectiveness\n- Historical healthcare utilization patterns\n\n**Environmental/Contextual Data:**\n- Seasonal patterns\n- Geographic risk factors\n- Social determinants of health\n- Sanitation and water quality\n- Vector presence (if applicable)\n\n**Intervention Data:**\n- Available prevention strategies\n- Control measure effectiveness\n- Resource availability\n- Public health infrastructure capacity\n\nCollecting and analyzing these data enables accurate risk stratification and targeted interventions."
        }
    ]

    return random.choice(templates)

def generate_surveillance_report():
    """Generate questions about creating surveillance reports"""
    disease = random.choice(DISEASES)
    week = random.randint(1, 52)
    year = random.randint(2022, 2024)

    templates = [
        {
            "question": f"Summarize the key findings for {disease} surveillance in week {week} of {year}.",
            "answer": f"**{disease.upper()} Surveillance Report**\n**Week {week}, {year}**\n\n**Executive Summary:**\nThis report summarizes {disease} surveillance activities and findings for epidemiological week {week} of {year}.\n\n**Case Count Summary:**\n- Total cases this week: {random.randint(100, 1000)}\n- Change from previous week: {random.choice(['+', '-'])}{random.randint(5, 30)}%\n- Year-to-date total: {random.randint(5000, 50000)}\n- Geographic distribution: {random.randint(15, 45)} states reporting\n\n**Epidemiological Trends:**\n- Incidence rate: {round(random.uniform(2.5, 15.5), 1)} per 100,000 population\n- Most affected regions: {', '.join(random.sample(REGIONS, 3))}\n- Most affected age group: {random.choice(DEMOGRAPHICS)}\n- Hospitalizations: {random.randint(50, 200)} this week\n\n**Laboratory Findings:**\n- Laboratory-confirmed cases: {random.randint(60, 95)}%\n- Predominant strain/serotype: [Varies by disease]\n- Test positivity rate: {random.randint(8, 25)}%\n\n**Public Health Actions:**\n- Enhanced surveillance in high-incidence areas\n- Targeted prevention messaging\n- Healthcare provider alerts issued\n- Outbreak investigations: {random.randint(1, 5)} active\n\n**Recommendations:**\n1. Continue routine surveillance\n2. Maintain prevention measures\n3. Monitor vulnerable populations\n4. Report unusual patterns immediately\n\n**Next Report:** Week {week + 1}, {year}"
        },
        {
            "question": f"What are the essential components of a disease surveillance report?",
            "answer": f"Essential components of a disease surveillance report:\n\n**1. Header Information:**\n- Disease name\n- Reporting period (dates, week, month)\n- Jurisdiction/geographic area\n- Report date and version\n- Reporting organization\n\n**2. Executive Summary:**\n- Key findings (2-3 bullet points)\n- Overall trend (increasing/decreasing/stable)\n- Public health significance\n- Action items\n\n**3. Case Data:**\n- Total case counts (confirmed, probable, suspect)\n- Demographic breakdown (age, sex, location)\n- Temporal trends (graphs/charts)\n- Geographic distribution (maps)\n- Comparison to baseline/previous periods\n\n**4. Epidemiological Analysis:**\n- Incidence and prevalence rates\n- Attack rates by subgroup\n- Mortality and morbidity statistics\n- Outbreak clusters\n- Risk factors identified\n\n**5. Laboratory Data:**\n- Testing volumes and positivity rates\n- Confirmed vs. clinical diagnoses\n- Strain/serotype information\n- Antimicrobial resistance patterns\n\n**6. Clinical Information:**\n- Symptom profiles\n- Severity distribution\n- Hospitalization rates\n- Complications and outcomes\n\n**7. Public Health Response:**\n- Interventions implemented\n- Investigation activities\n- Control measures\n- Communication efforts\n\n**8. Interpretation and Recommendations:**\n- Significance of findings\n- Implications for public health\n- Recommended actions\n- Future surveillance priorities\n\n**9. Methods and Limitations:**\n- Data sources\n- Case definitions\n- Surveillance methods\n- Data quality notes\n- Known limitations\n\n**10. Appendices:**\n- Detailed tables\n- Supplementary figures\n- Technical notes\n- Contact information"
        }
    ]

    return random.choice(templates)

def generate_vaccination_coverage():
    """Generate questions about vaccination coverage and surveillance"""
    disease = random.choice(["measles", "influenza", "COVID-19", "HPV", "hepatitis B", "pneumococcal disease"])
    coverage = random.randint(40, 85)
    target = random.randint(85, 95)

    templates = [
        {
            "question": f"Current {disease} vaccination coverage is {coverage}% in our jurisdiction. The target is {target}%. What are the surveillance implications?",
            "answer": f"**Vaccination Coverage Analysis for {disease}**\n\n**Current Status:**\n- Coverage: {coverage}%\n- Target: {target}%\n- Gap: {target - coverage} percentage points\n\n**Surveillance Implications:**\n\n**1. Disease Risk:**\n- **Below herd immunity threshold** - Population remains vulnerable\n- **{100 - coverage}% of population susceptible** to infection\n- **Outbreak risk is ELEVATED** due to coverage gap\n- Clustering of unvaccinated individuals increases local risk\n\n**2. Enhanced Surveillance Needs:**\n- **Intensified case detection** in undervaccinated communities\n- **Monitoring of zero-dose and under-vaccinated cohorts**\n- **Geographic mapping** of coverage gaps\n- **Tracking breakthrough infections** in vaccinated populations\n- **Surveillance for vaccine-preventable disease resurgence**\n\n**3. Outbreak Response Preparedness:**\n- High likelihood of outbreaks in low-coverage pockets\n- Need for rapid response protocols\n- Ring vaccination strategies for outbreak containment\n- Healthcare surge capacity planning\n\n**4. Equity Considerations:**\n- Identify underserved populations with low coverage\n- Address barriers to vaccination access\n- Targeted outreach to high-risk, low-coverage groups\n\n**Recommended Actions:**\n1. **Immediate:** Map coverage by sub-jurisdiction to identify gaps\n2. **Short-term:** Implement catch-up vaccination campaigns\n3. **Ongoing:** Enhanced surveillance in low-coverage areas\n4. **Long-term:** Address systemic barriers to vaccination\n\n**Risk Timeline:**\n- Without improvement: Outbreak likely within 6-12 months\n- With targeted campaigns: Risk reduction in 3-6 months\n\n**Monitoring Metrics:**\n- Weekly vaccination uptake rates\n- Coverage by demographic subgroups\n- Geographic coverage disparities\n- Zero-dose and under-vaccinated tracking"
        },
        {
            "question": f"How do we monitor vaccination program effectiveness through surveillance?",
            "answer": f"**Monitoring Vaccination Program Effectiveness Through Surveillance:**\n\n**1. Coverage Monitoring:**\n- **Vaccination rates** by age, geography, demographics\n- **Timeliness** of vaccination (on-schedule vs. delayed)\n- **Dose completion rates** (full series vs. partial)\n- **Equity metrics** across populations\n- **Trend analysis** over time\n\n**2. Disease Incidence Surveillance:**\n- **Case counts** in vaccinated vs. unvaccinated populations\n- **Attack rates** by vaccination status\n- **Outbreak frequency and size** in vaccinated populations\n- **Geographic correlation** between coverage and disease incidence\n- **Temporal trends** following program implementation\n\n**3. Vaccine Effectiveness (VE) Studies:**\n- **Test-negative design** studies\n- **Cohort studies** comparing vaccinated vs. unvaccinated\n- **Case-control studies** in outbreak settings\n- **VE by age group, time since vaccination, and strain**\n- **Breakthrough infection analysis**\n\n**4. Safety Surveillance:**\n- **Adverse events following immunization (AEFI)**\n- **Signal detection** for rare adverse events\n- **Risk-benefit monitoring**\n- **Vaccine product quality surveillance**\n\n**5. Immunological Surveillance:**\n- **Seroprevalence studies** to assess population immunity\n- **Antibody levels** over time (waning immunity)\n- **Immunogenicity** in special populations\n- **Correlates of protection**\n\n**6. Healthcare Utilization:**\n- **Hospitalizations** prevented\n- **Emergency department visits** for vaccine-preventable diseases\n- **Intensive care admissions**\n- **Mortality rates**\n\n**7. Program Process Indicators:**\n- **Vaccine supply and distribution**\n- **Cold chain compliance**\n- **Wastage rates**\n- **Provider participation**\n- **Recall/reminder system effectiveness**\n\n**Key Performance Indicators:**\n- Reduction in disease incidence post-program\n- Decrease in outbreaks in high-coverage areas\n- Decline in hospitalizations and deaths\n- Improved health equity metrics\n- High vaccine effectiveness estimates\n\n**Data Integration:**\n- Link immunization registries with disease surveillance\n- Integrate laboratory and clinical data\n- Real-time monitoring dashboards\n- Regular evaluation reports\n\n**Continuous Improvement:**\n- Regular program evaluation\n- Adaptive strategies based on surveillance findings\n- Stakeholder feedback integration\n- Evidence-based policy adjustments"
        }
    ]

    return random.choice(templates)

def generate_data_interpretation():
    """Generate questions about interpreting surveillance data"""
    disease = random.choice(DISEASES)

    templates = [
        {
            "question": f"What does an R₀ (basic reproduction number) of {round(random.uniform(1.5, 4.0), 1)} mean for {disease} transmission?",
            "answer": f"**Interpreting R₀ = {round(random.uniform(1.5, 4.0), 1)} for {disease}:**\n\n**Definition:**\nR₀ (basic reproduction number) represents the average number of secondary infections caused by one infected individual in a completely susceptible population with no interventions.\n\n**Implications:**\n\n**1. Transmission Dynamics:**\n- **R₀ > 1:** Disease will spread in the population (epidemic growth)\n- **Each case generates ~{round(random.uniform(1.5, 4.0), 1)} new cases** on average\n- **Exponential growth** potential without intervention\n\n**2. Herd Immunity Threshold:**\n- Herd immunity threshold = (1 - 1/R₀) × 100\n- For this R₀: ~{round((1 - 1/random.uniform(1.5, 4.0)) * 100)}% of population needs immunity\n- This can be achieved through vaccination or natural infection\n\n**3. Control Measure Requirements:**\n- Need to **reduce effective reproduction number (Rₑ) below 1**\n- Interventions must reduce transmission by >{round((1 - 1/random.uniform(1.5, 4.0)) * 100)}%\n- Multiple layered interventions likely needed\n\n**4. Outbreak Potential:**\n- **Moderate to high** transmission potential\n- Rapid spread expected in susceptible populations\n- Clusters can quickly become large outbreaks\n- Early intervention critical\n\n**5. Surveillance Priorities:**\n- **Early detection essential** - exponential growth is rapid\n- **Contact tracing** must be swift and comprehensive\n- **Case isolation** reduces effective R\n- **Quarantine of contacts** breaks transmission chains\n\n**Control Strategies to Reduce R₀:**\n- Vaccination (if available)\n- Social distancing measures\n- Isolation and quarantine\n- Personal protective equipment\n- Environmental modifications\n- Behavioral interventions\n\n**Monitoring Effectiveness:**\n- Track effective reproduction number (Rₑ) over time\n- Goal: Reduce Rₑ < 1 to stop epidemic growth\n- Regular re-estimation as interventions are implemented"
        },
        {
            "question": f"Explain what it means when {disease} incidence increases from 5 to 15 cases per 100,000 population.",
            "answer": f"**Interpreting Incidence Increase: {disease}**\n**From 5 to 15 per 100,000 population**\n\n**Quantitative Interpretation:**\n\n**Absolute Change:**\n- Increase of 10 cases per 100,000 population\n- In a city of 1 million: 50 → 150 cases (+100 cases)\n\n**Relative Change:**\n- **200% increase** (3-fold rise)\n- **Tripling of disease burden**\n\n**Public Health Significance:**\n\n**1. Epidemic Threshold:**\n- Whether this constitutes an \"outbreak\" depends on:\n  - Historical baseline for this disease\n  - Seasonal patterns\n  - Expected variation\n- A 200% increase typically **warrants investigation**\n\n**2. Healthcare Impact:**\n- **Increased healthcare utilization**\n  - More clinic/ED visits\n  - Potential hospitalizations\n  - Diagnostic testing demands\n- **Resource allocation needs** may shift\n\n**3. Transmission Assessment:**\n- Suggests **ongoing active transmission** in community\n- May indicate:\n  - New introduction of pathogen\n  - Increased exposure events\n  - Reduced population immunity\n  - Environmental or behavioral changes\n\n**4. Surveillance Actions:**\n- **Immediate:** Verify data quality and completeness\n- **Investigation:** Conduct outbreak investigation if indicated\n- **Enhanced monitoring:** Increase surveillance frequency\n- **Geographic analysis:** Identify high-incidence areas\n- **Demographic analysis:** Identify high-risk groups\n\n**5. Risk Communication:**\n- **Public notification** may be appropriate\n- **Healthcare provider alerts**\n- **Prevention messaging** to at-risk populations\n\n**Contextual Considerations:**\n- **Baseline rates:** 5 per 100,000 is relatively low\n- **15 per 100,000:** Still moderate in absolute terms\n- **Trend direction:** More important than single point\n- **Comparison:** Compare to other jurisdictions, previous years\n\n**Next Steps:**\n1. Confirm case definitions and data accuracy\n2. Analyze temporal trends (is it still rising?)\n3. Geographic and demographic sub-analysis\n4. Identify potential sources/exposures\n5. Implement control measures if indicated\n6. Continue monitoring with enhanced surveillance\n\n**Decision Threshold:**\n- A 200% increase generally triggers public health response\n- Response intensity depends on disease severity and transmission potential"
        }
    ]

    return random.choice(templates)

def generate_syndromic_surveillance():
    """Generate questions about syndromic surveillance"""
    syndrome = random.choice(["influenza-like illness", "acute gastroenteritis", "acute respiratory illness",
                              "fever and rash", "acute flaccid paralysis", "hemorrhagic fever"])

    templates = [
        {
            "question": f"How is syndromic surveillance used to detect {syndrome} outbreaks early?",
            "answer": f"**Syndromic Surveillance for {syndrome}:**\n\n**Definition:**\nSyndromic surveillance monitors health indicators in real-time before laboratory confirmation, enabling early outbreak detection.\n\n**Data Sources for {syndrome}:**\n\n**1. Healthcare Data:**\n- **Emergency department visits** with relevant chief complaints\n- **Ambulatory care visits** for syndrome-related symptoms\n- **Hospital admissions** with syndrome diagnosis\n- **Telemedicine consultations**\n- **EMS/911 calls** with relevant symptoms\n\n**2. Over-the-Counter (OTC) Sales:**\n- Fever reducers and pain relievers\n- Anti-diarrheal medications (for GI syndromes)\n- Cough and cold medications (for respiratory syndromes)\n- Sales spikes can precede clinical presentation\n\n**3. School and Workplace Absenteeism:**\n- School nurse visits\n- Employee sick leave patterns\n- Daycare illness reports\n\n**4. Laboratory Orders:**\n- Diagnostic test orders (before results available)\n- Increases in relevant test requests\n\n**5. Digital Data:**\n- Search engine queries for symptoms\n- Social media mentions of illness\n- Symptom-checking app data\n\n**Early Detection Mechanisms:**\n\n**1. Aberration Detection:**\n- Statistical algorithms detect deviations from baseline\n- Time-series analysis identifies unusual patterns\n- Threshold alerts for predefined levels\n\n**2. Temporal Monitoring:**\n- Real-time or near-real-time data flow\n- Daily or even hourly updates\n- Trend analysis over hours/days vs. weeks\n\n**3. Geographic Hotspot Detection:**\n- Spatial clustering algorithms\n- GIS mapping of syndrome occurrences\n- Identification of geographic outbreak foci\n\n**4. Multi-source Integration:**\n- Combining multiple data streams\n- Cross-validation of signals\n- Reduced false positives through correlation\n\n**Advantages of Syndromic Surveillance:**\n- **Timeliness:** Days to weeks earlier than traditional surveillance\n- **Sensitivity:** Detects outbreaks before etiology confirmed\n- **Broad coverage:** Captures cases not seeking care or tested\n- **Early warning:** Enables rapid public health response\n- **Resource allocation:** Guides surge capacity planning\n\n**Limitations:**\n- **Lower specificity:** Many causes of similar syndromes\n- **False positives:** Non-outbreak increases (e.g., seasonal patterns)\n- **Data quality:** Depends on accurate coding and reporting\n- **Interpretation challenges:** Requires expertise to assess signals\n\n**Response Protocol for Syndrome Increase:**\n\n**1. Signal Detection (Automated):**\n- Algorithm flags statistical aberration\n- Alert generated for public health staff\n\n**2. Signal Validation:**\n- Review data for errors or artifacts\n- Check for known explanations (events, seasonality)\n- Assess magnitude and trend\n\n**3. Investigation Trigger:**\n- If signal credible, initiate investigation\n- Enhance traditional surveillance\n- Conduct field investigation if needed\n\n**4. Laboratory Confirmation:**\n- Collect specimens from suspected cases\n- Identify etiology\n- Confirm outbreak\n\n**5. Public Health Action:**\n- Implement control measures\n- Risk communication\n- Continue monitoring\n\n**Example Application for {syndrome}:**\n- Monitor syndrome-specific indicators daily\n- Establish dynamic baselines accounting for seasonality\n- Use multiple complementary data sources\n- Integrate with laboratory surveillance once etiology identified\n- Enable 3-7 day earlier outbreak detection compared to traditional methods\n\n**Key Success Factors:**\n- Automated, real-time data systems\n- Clear case definitions and algorithms\n- Trained epidemiologists for signal interpretation\n- Integrated response protocols\n- Strong partnerships with healthcare and data providers"
        }
    ]

    return random.choice(templates)

def generate_contact_tracing():
    """Generate questions about contact tracing and surveillance"""
    disease = random.choice(["COVID-19", "measles", "tuberculosis", "Ebola", "meningitis"])

    templates = [
        {
            "question": f"What is the contact tracing protocol for {disease} and how does it support surveillance?",
            "answer": f"**Contact Tracing Protocol for {disease}:**\n\n**Purpose:**\nContact tracing identifies, assesses, and manages individuals exposed to {disease} to prevent further transmission and provide surveillance data.\n\n**Step 1: Case Investigation**\n- Interview confirmed case about:\n  - Symptom onset date\n  - Infectious period (disease-specific)\n  - Locations visited during infectious period\n  - Close contacts during infectious period\n  - Potential exposure sources\n\n**Step 2: Contact Identification**\n\n**Close Contact Definition (disease-specific):**\nFor {disease}, close contacts typically include:\n- Household members\n- People within 6 feet for ≥15 minutes (respiratory diseases)\n- People with direct contact with infectious material\n- Healthcare workers without appropriate PPE\n- [Disease-specific criteria apply]\n\n**Contact List Development:**\n- Name, contact information\n- Relationship to case\n- Date and duration of exposure\n- Type of exposure (high-risk vs. low-risk)\n- Current symptoms status\n\n**Step 3: Contact Notification**\n- **Rapid notification** (within 24-48 hours of case identification)\n- Inform about exposure without revealing case identity\n- Explain disease, transmission, symptoms\n- Provide guidance on monitoring and testing\n\n**Step 4: Contact Management**\n\n**Monitoring:**\n- Daily symptom checks during incubation period\n- Temperature monitoring if relevant\n- Immediate reporting of symptoms\n\n**Quarantine/Isolation Guidelines:**\n- **High-risk contacts:** May require quarantine\n- Duration based on incubation period\n- Criteria for release from quarantine\n- Support for adherence (food, medical care, financial)\n\n**Testing:**\n- Immediate testing of symptomatic contacts\n- Serial testing of high-risk asymptomatic contacts (if applicable)\n- Post-exposure testing timeline\n\n**Prophylaxis:**\n- Post-exposure prophylaxis if available (e.g., vaccines, medications)\n- Timing and eligibility criteria\n\n**Step 5: Data Management and Surveillance Integration**\n\n**Surveillance Contributions:**\n\n**1. Transmission Chain Mapping:**\n- Identify source of infection\n- Map secondary cases from primary case\n- Calculate secondary attack rates\n- Visualize transmission networks\n- Identify super-spreading events\n\n**2. Epidemiological Parameters:**\n- **Serial interval:** Time between symptom onset in successive cases\n- **Incubation period:** Distribution of incubation times\n- **Generation time:** Time between infections in transmission chain\n- **Reproduction number (R):** Average secondary cases per case\n\n**3. Risk Factor Identification:**\n- High-risk exposure types\n- High-risk settings (households, workplaces, events)\n- Protective factors (vaccination, PPE, behaviors)\n- Vulnerable populations\n\n**4. Outbreak Control Assessment:**\n- Proportion of contacts successfully traced\n- Speed of tracing (time from case ID to contact notification)\n- Quarantine compliance rates\n- Secondary case prevention\n\n**5. Early Warning:**\n- Detect chains of transmission\n- Identify emerging clusters\n- Geographic spread patterns\n- Populations at risk\n\n**Performance Metrics:**\n\n**Timeliness:**\n- Time from symptom onset to case isolation\n- Time from case ID to contact notification\n- **Target: <24 hours for rapid response**\n\n**Completeness:**\n- Percentage of contacts identified and reached\n- **Target: ≥80% of contacts traced**\n\n**Effectiveness:**\n- Secondary attack rate in traced vs. untraced contacts\n- Reduction in onward transmission\n- Epidemic curve impact\n\n**Data Systems:**\n- Electronic case investigation forms\n- Contact management databases\n- Integration with disease surveillance systems\n- Real-time data dashboards\n- Secure data sharing platforms\n\n**Challenges:**\n- Incomplete contact recall by cases\n- Anonymous or untraceable contacts\n- Stigma and non-cooperation\n- Resource intensity\n- Privacy concerns\n- Large outbreak surge capacity\n\n**Best Practices:**\n- Rapid, respectful, confidential approach\n- Cultural and linguistic competency\n- Community engagement and trust\n- Adequate staffing and training\n- Technology-enabled efficiency (apps, databases)\n- Clear protocols and supervision\n- Support for cases and contacts\n\n**Surveillance Integration:**\nContact tracing data feeds directly into surveillance systems, providing:\n- Real-time transmission intelligence\n- Detailed epidemiological insights\n- Evidence base for control measures\n- Evaluation of intervention effectiveness\n\nEffective contact tracing is surveillance in action, converting reactive case finding into proactive transmission interruption."
        }
    ]

    return random.choice(templates)

def generate_zoonotic_surveillance():
    """Generate questions about zoonotic disease surveillance"""
    disease = random.choice(["rabies", "avian influenza", "West Nile virus", "Lyme disease", "hantavirus", "plague"])
    animal = random.choice(["wildlife", "livestock", "domestic animals", "rodents", "birds", "mosquitoes", "ticks"])

    templates = [
        {
            "question": f"How does surveillance of {disease} in {animal} populations inform human public health?",
            "answer": f"**Zoonotic Surveillance: {disease} in {animal}**\n\n**One Health Approach:**\nIntegrated surveillance of {disease} in {animal} populations provides critical early warning for human health risks.\n\n**Surveillance Components:**\n\n**1. Animal Surveillance:**\n- **Passive surveillance:** Sick/dead {animal} reports\n- **Active surveillance:** Systematic sampling and testing\n- **Sentinel surveillance:** High-risk animals monitored regularly\n- **Vector surveillance:** If vector-borne (mosquitoes, ticks)\n- **Environmental surveillance:** Pathogen in environment\n\n**2. Laboratory Testing:**\n- Diagnostic testing of animal specimens\n- Pathogen detection and characterization\n- Strain typing and genomic sequencing\n- Antimicrobial resistance testing\n- Comparison with human isolates\n\n**3. Geographic and Temporal Mapping:**\n- GIS mapping of positive animals\n- Identification of high-risk areas\n- Seasonal patterns in animal populations\n- Expansion of geographic range\n\n**Public Health Benefits:**\n\n**1. Early Warning System:**\n- **Animal cases often precede human cases** by weeks to months\n- Enables proactive rather than reactive response\n- Time to implement preventive measures\n- Risk communication to at-risk populations\n\n**Example for {disease}:**\n- Detection in {animal} indicates human exposure risk\n- Geographic areas with positive {animal} are high-risk for humans\n- Temporal patterns predict human disease season\n\n**2. Risk Assessment:**\n- Quantify environmental risk to humans\n- Identify high-risk occupations (veterinarians, farmers, outdoor workers)\n- Inform personal protective recommendations\n- Guide allocation of prevention resources\n\n**3. Targeted Prevention:**\n- **Vector control:** Insecticide spraying in high-risk areas\n- **Animal vaccination:** Reduce reservoir (e.g., wildlife rabies vaccination)\n- **Public advisories:** Warnings about exposure risks\n- **Occupational safety:** Enhanced PPE recommendations\n- **Avoid exposure:** Guidance on avoiding contact with infected animals/vectors\n\n**4. Outbreak Investigation:**\n- Identify animal source of human outbreaks\n- Trace exposure events\n- Guide outbreak control measures\n- Prevent additional human cases\n\n**5. Monitoring Intervention Effectiveness:**\n- Track impact of animal vaccination programs\n- Assess vector control effectiveness\n- Evaluate environmental interventions\n- Adjust strategies based on surveillance data\n\n**6. Pathogen Evolution Tracking:**\n- Monitor for novel strains in animal reservoirs\n- Detect mutations increasing human transmission risk\n- Identify antimicrobial resistance emergence\n- Pandemic preparedness (e.g., novel influenza strains)\n\n**Integrated Surveillance Activities:**\n\n**Collaboration:**\n- **Public health + veterinary medicine + environmental science**\n- Wildlife biologists, agricultural agencies, vector control districts\n- Coordinated data sharing and response\n\n**Data Integration:**\n- Combine human, animal, and environmental data\n- Correlate animal positivity with human case rates\n- Predictive modeling of human risk based on animal surveillance\n\n**Risk Communication:**\n- Real-time updates to healthcare providers\n- Public advisories based on animal surveillance findings\n- Targeted messaging to high-risk groups\n\n**Case Study Example - {disease}:**\n1. **Animal surveillance** detects {disease} in {animal} in Region A\n2. **Risk assessment:** High human exposure risk identified\n3. **Alert issued:** Public and healthcare providers notified\n4. **Prevention activated:** Personal protection guidance, vector control\n5. **Enhanced human surveillance:** Increased clinical awareness and testing\n6. **Early detection:** Human cases detected and treated earlier\n7. **Outcome:** Reduced human morbidity and mortality\n\n**Key Performance Indicators:**\n- Lead time: Days/weeks between animal and human case detection\n- Geographic concordance: Overlap of animal and human cases\n- Sensitivity: Proportion of human risk areas predicted by animal surveillance\n- Public health impact: Human cases prevented through early warning\n\n**Challenges:**\n- Resource intensive (both human and animal surveillance)\n- Requires interdisciplinary collaboration\n- Data integration across sectors\n- Interpretation of animal findings for human risk\n- Political and organizational boundaries\n\n**Future Directions:**\n- Enhanced genomic surveillance (pathogen evolution)\n- Modeling to predict spillover events\n- Climate and environmental monitoring integration\n- Real-time data sharing platforms\n- Strengthened One Health infrastructure\n\n**Conclusion:**\nAnimal surveillance for {disease} is not separate from human public health—it's an integral component. Early detection in {animal} populations provides actionable intelligence to protect human health, prevent outbreaks, and guide resource allocation. The One Health approach recognizes that human, animal, and environmental health are interconnected and must be addressed holistically."
        }
    ]

    return random.choice(templates)

def generate_global_surveillance():
    """Generate questions about global surveillance and emerging threats"""
    disease = random.choice(DISEASES)
    country = random.choice(["Nigeria", "India", "Brazil", "China", "South Africa", "Indonesia", "Kenya", "Vietnam"])

    templates = [
        {
            "question": f"An outbreak of {disease} has been reported in {country}. What are the global surveillance implications?",
            "answer": f"**Global Surveillance Response: {disease} Outbreak in {country}**\n\n**Immediate Assessment:**\n\n**1. International Health Regulations (IHR) Notification:**\n- {country} must assess if outbreak meets IHR criteria\n- WHO notification required if potential public health emergency of international concern (PHEIC)\n- Immediate information sharing through IHR event management system\n\n**2. Risk Assessment for International Spread:**\n- **Travel connectivity:** {country}'s international travel volume and routes\n- **Incubation period:** Can infected travelers spread disease internationally?\n- **Transmissibility:** Human-to-human transmission potential\n- **Severity:** Morbidity and mortality rates\n- **Control capacity:** {country}'s outbreak control capabilities\n\n**Global Surveillance Actions:**\n\n**1. Enhanced Border Surveillance:**\n\n**Entry Screening:**\n- **Countries with direct travel from {country}:** Implement arrival screening\n- Symptom assessment questionnaires\n- Temperature screening if relevant\n- Travel history collection (past 21 days)\n- Contact information for follow-up\n\n**Exit Screening (if implemented in {country}):**\n- Pre-departure health screening\n- Prevents symptomatic travelers from boarding\n- More effective than entry screening alone\n\n**2. Healthcare System Alerting:**\n\n**Clinical Awareness:**\n- Alerts to emergency departments, primary care providers\n- Index of suspicion for patients with:\n  - Compatible symptoms\n  - Travel history to {country}\n  - Contact with travelers from {country}\n\n**Diagnostic Preparedness:**\n- Ensure testing capacity available\n- Expedited testing protocols for suspected cases\n- Laboratory biosafety protocols reviewed\n- Specimen shipping arrangements confirmed\n\n**3. Enhanced Surveillance in Receiving Countries:**\n\n**Indicator-Based Surveillance:**\n- Review of emergency department visits for compatible syndromes\n- Laboratory surveillance for similar pathogens\n- Syndromic surveillance intensification\n- Hospitalization monitoring\n\n**Event-Based Surveillance:**\n- Media monitoring for unofficial reports\n- Healthcare rumor surveillance\n- Community reporting mechanisms\n\n**4. International Collaboration:**\n\n**WHO Coordination:**\n- Situation reports and risk assessments\n- Technical guidance for countries\n- Coordination of international response\n- Resource mobilization if needed\n\n**Bilateral Communication:**\n- Countries with high travel volumes from {country}\n- Regional health networks activation\n- Information sharing on cases, risk factors, control measures\n\n**Laboratory Networks:**\n- WHO reference laboratories engaged\n- Pathogen characterization and sequencing\n- Diagnostic assay sharing\n- Quality assurance for testing\n\n**5. Traveler Health Measures:**\n\n**Travel Advisories:**\n- CDC/WHO travel notices issued (Level 1, 2, or 3)\n- Guidance for travelers to {country}\n- Guidance for travelers returning from {country}\n\n**Pre-Travel:**\n- Vaccination if available and recommended\n- Preventive medication if applicable\n- Risk reduction counseling\n\n**During Travel:**\n- Behavior recommendations to reduce risk\n- Recognition of symptoms\n- Access to medical care if symptomatic\n\n**Post-Travel:**\n- Self-monitoring for symptoms (duration based on incubation period)\n- Immediate healthcare seeking if symptoms develop\n- Notification of travel history to healthcare providers\n\n**6. Public Health Preparedness:**\n\n**Response Planning:**\n- Review and activate imported case protocols\n- Isolation and infection control readiness\n- Contact tracing capacity assessment\n- Surge capacity planning\n\n**Risk Communication:**\n- Public information on disease and risks\n- Countering misinformation\n- Updates as situation evolves\n\n**7. Epidemiological Intelligence:**\n\n**Monitoring Outbreak Evolution:**\n- Daily situation reports from {country} and WHO\n- Tracking case counts, geographic spread, mortality\n- Assessment of control measure effectiveness\n- Modeling of international spread risk\n\n**Genomic Surveillance:**\n- Sequencing of cases (in {country} and internationally)\n- Tracking variants and evolution\n- Understanding transmission pathways\n- Informing diagnostics and treatments\n\n**Special Considerations for {disease}:**\n[Disease-specific considerations, e.g.,]\n- Incubation period: [X] days → monitoring period for travelers\n- Transmission: [route] → specific precautions\n- High-risk groups: [populations] → targeted surveillance\n- Vaccine availability: [yes/no] → vaccination campaigns\n\n**Escalation Scenarios:**\n\n**Scenario 1: Contained Outbreak**\n- Outbreak limited to {country}\n- No international spread detected\n- Maintain surveillance vigilance\n- Support {country}'s response\n- De-escalate measures as outbreak controlled\n\n**Scenario 2: Limited International Spread**\n- Sporadic imported cases in other countries\n- No secondary transmission internationally\n- Continue enhanced surveillance\n- Manage imported cases with isolation/treatment\n- Continue monitoring\n\n**Scenario 3: Sustained International Transmission**\n- Secondary cases in multiple countries\n- Potential PHEIC declaration\n- Intensified international response\n- Possible travel restrictions\n- WHO Emergency Committee convened\n\n**Lessons from Recent Outbreaks:**\n- Early international coordination critical\n- Transparency in reporting builds trust\n- Genetic sequencing essential for tracking\n- Travel screening has limited effectiveness (most cases asymptomatic during travel)\n- Contact tracing and isolation most effective\n- Risk communication must be clear, consistent, evidence-based\n\n**Key Performance Indicators:**\n- Time from outbreak report to international alert: <24 hours\n- Proportion of countries implementing enhanced surveillance: >80% of connected countries\n- Time to detection of imported cases: <48 hours of symptom onset\n- Laboratory confirmation turnaround time: <24 hours\n\n**Conclusion:**\nA {disease} outbreak in {country} requires immediate, coordinated global surveillance response. The interconnected nature of modern travel means no outbreak is purely local. Effective global surveillance combines early detection, rapid communication, coordinated response, and continuous monitoring to prevent or minimize international spread and protect global health security."
        }
    ]

    return random.choice(templates)

# -----------------------------------------------------------------------------
# Main Dataset Generation Function
# -----------------------------------------------------------------------------

def generate_dataset(num_examples=5000, train_split=0.8, val_split=0.1):
    """
    Generate comprehensive public health surveillance dataset.

    Args:
        num_examples: Total number of examples to generate
        train_split: Proportion for training set
        val_split: Proportion for validation set (remainder is test)
    """

    print(f"Generating {num_examples} public health surveillance examples...")

    # Generation functions and their weights (to ensure balanced dataset)
    generators = [
        (generate_outbreak_detection, 0.15),
        (generate_trend_analysis, 0.15),
        (generate_risk_assessment, 0.15),
        (generate_surveillance_report, 0.10),
        (generate_vaccination_coverage, 0.10),
        (generate_data_interpretation, 0.10),
        (generate_syndromic_surveillance, 0.10),
        (generate_contact_tracing, 0.05),
        (generate_zoonotic_surveillance, 0.05),
        (generate_global_surveillance, 0.05),
    ]

    # Calculate number of examples per category
    examples_per_category = [(gen, int(num_examples * weight)) for gen, weight in generators]

    # Generate all examples
    all_examples = []
    for generator, count in examples_per_category:
        category_name = generator.__name__.replace("generate_", "")
        print(f"  Generating {count} {category_name} examples...")
        for _ in range(count):
            example = generator()
            example["category"] = category_name
            all_examples.append(example)

    # Shuffle all examples
    random.shuffle(all_examples)

    # Convert to conversation format for nanochat
    conversations = []
    for ex in all_examples:
        conversation = {
            "messages": [
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]}
            ],
            "metadata": {
                "category": ex["category"],
                "domain": "public_health_surveillance"
            }
        }
        conversations.append(conversation)

    # Split into train/val/test
    train_size = int(len(conversations) * train_split)
    val_size = int(len(conversations) * val_split)

    train_data = conversations[:train_size]
    val_data = conversations[train_size:train_size + val_size]
    test_data = conversations[train_size + val_size:]

    print(f"\nDataset splits:")
    print(f"  Training: {len(train_data)} examples")
    print(f"  Validation: {len(val_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    print(f"  Total: {len(conversations)} examples")

    return train_data, val_data, test_data

def save_dataset(train_data, val_data, test_data, output_dir="data/surveillance"):
    """Save datasets to JSON files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save training data
    train_file = output_path / "train.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved training data to {train_file}")

    # Save validation data
    val_file = output_path / "validation.json"
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"Saved validation data to {val_file}")

    # Save test data
    test_file = output_path / "test.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"Saved test data to {test_file}")

    # Save dataset statistics
    stats = {
        "total_examples": len(train_data) + len(val_data) + len(test_data),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "generated_date": datetime.now().isoformat(),
        "categories": {
            "outbreak_detection": "Questions about detecting disease outbreaks",
            "trend_analysis": "Analyzing epidemiological trends",
            "risk_assessment": "Public health risk assessments",
            "surveillance_report": "Creating surveillance reports",
            "vaccination_coverage": "Vaccination program surveillance",
            "data_interpretation": "Interpreting surveillance data and metrics",
            "syndromic_surveillance": "Syndromic surveillance systems",
            "contact_tracing": "Contact tracing protocols",
            "zoonotic_surveillance": "Animal-human disease surveillance",
            "global_surveillance": "International surveillance coordination"
        }
    }

    stats_file = output_path / "dataset_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved dataset statistics to {stats_file}")

    print(f"\nDataset generation complete!")
    print(f"   Total conversations: {stats['total_examples']}")
    print(f"   Location: {output_path.absolute()}")

def main():
    """Main function to generate and save surveillance dataset."""

    # Generate dataset (5000 examples by default, adjust as needed)
    train_data, val_data, test_data = generate_dataset(
        num_examples=5000,
        train_split=0.8,
        val_split=0.1
    )

    # Save to files
    save_dataset(train_data, val_data, test_data)

    # Print sample
    print("\n" + "="*80)
    print("SAMPLE CONVERSATION:")
    print("="*80)
    sample = train_data[0]
    print(f"\nCategory: {sample['metadata']['category']}")
    print(f"\nUser: {sample['messages'][0]['content'][:200]}...")
    print(f"\nAssistant: {sample['messages'][1]['content'][:400]}...")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
