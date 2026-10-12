"""
变量语义库。

选题二的核心假设是：语言模型只在**有语义**时才提供有用的因果先验。
因此每个网络都需要两份东西：

1. 变量名 -> 真实含义（真实名模式使用）
2. 匿名映射（匿名模式使用，此时不提供任何含义）

含义文本来自各 bnlearn 网络的原始变量说明，保持简短、客观，
不包含任何关于"是否存在边"的提示，避免把答案泄漏给模型。
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


# ============================================================
# 网络所属领域（真实名模式下放进 prompt 的背景信息）
# ============================================================

NETWORK_DOMAIN = {
    "ALARM": "ALARM, a medical monitoring Bayesian network for patient ventilation",
    "ASIA": "ASIA, a medical Bayesian network for lung disease diagnosis",
    "CHILD": "CHILD, a neonatal intensive care Bayesian network",
    "INSURANCE": "INSURANCE, a car insurance risk Bayesian network",
    "HAILFINDER": "HAILFINDER, a thunderstorm nowcasting Bayesian network",
}


# ============================================================
# 变量含义
# ============================================================

VARIABLE_MEANINGS: dict[str, dict[str, str]] = {
    "ALARM": {
        "HISTORY": "the patient has a history of serious illness",
        "CVP": "central venous pressure, the blood pressure in the vena cava",
        "PCWP": "pulmonary capillary wedge pressure, an indirect measure of left atrial pressure",
        "HYPOVOLEMIA": "hypovolemia, abnormally low blood volume",
        "LVEDVOLUME": "left ventricular end-diastolic volume, how full the left ventricle is before it contracts",
        "LVFAILURE": "left ventricular failure, the left side of the heart pumps inadequately",
        "STROKEVOLUME": "stroke volume, the amount of blood ejected by the heart per beat",
        "ERRLOWOUTPUT": "measurement error: the monitor reports an erroneously low output",
        "HRBP": "heart rate as measured from the blood pressure signal",
        "HREKG": "heart rate as measured from the electrocardiogram signal",
        "ERRCAUTER": "measurement error: cauterization caused an erroneous reading",
        "HRSAT": "heart rate as measured from the oxygen saturation signal",
        "INSUFFANESTH": "insufficient anesthesia, the patient is not deeply anesthetized enough",
        "ANAPHYLAXIS": "anaphylaxis, a severe systemic allergic reaction",
        "TPR": "total peripheral resistance, the resistance of the systemic blood vessels",
        "EXPCO2": "expired carbon dioxide level",
        "KINKEDTUBE": "the endotracheal tube is kinked",
        "MINVOL": "minute ventilation, the volume of air breathed per minute",
        "FIO2": "fraction of inspired oxygen delivered by the ventilator",
        "PVSAT": "pulmonary venous oxygen saturation",
        "SAO2": "arterial oxygen saturation",
        "PAP": "pulmonary artery pressure",
        "PULMEMBOLUS": "pulmonary embolism, a blockage in the pulmonary artery",
        "SHUNT": "intrapulmonary shunt, blood passing through the lung without being oxygenated",
        "INTUBATION": "the patient has been intubated with an endotracheal tube",
        "PRESS": "airway pressure measured by the ventilator",
        "DISCONNECT": "the breathing circuit has become disconnected",
        "MINVOLSET": "the minute ventilation set on the ventilator",
        "VENTMACH": "the ventilator machine that drives breathing",
        "VENTTUBE": "the endotracheal tube through which gas is delivered",
        "VENTLUNG": "the state of the lung as ventilated",
        "VENTALV": "the state of the alveoli as ventilated",
        "ARTCO2": "arterial carbon dioxide level",
        "CATECHOL": "catecholamine level, stress hormones in the blood",
        "HR": "heart rate, the number of heartbeats per minute",
        "CO": "cardiac output, the volume of blood pumped by the heart per minute",
        "BP": "blood pressure, the arterial pressure measured in the patient",
    },
    "ASIA": {
        "asia": "the patient has recently visited Asia",
        "tub": "the patient has tuberculosis",
        "smoke": "the patient smokes",
        "lung": "the patient has lung cancer",
        "bronc": "the patient has bronchitis",
        "either": "the patient has either tuberculosis or lung cancer",
        "xray": "the chest X-ray result is abnormal",
        "dysp": "the patient suffers from dyspnea, difficulty in breathing",
    },
    "CHILD": {
        "BirthAsphyxia": "the infant suffered birth asphyxia, oxygen deprivation during birth",
        "HypDistrib": "the desaturation is distributed unevenly between upper and lower body",
        "HypoxiaInO2": "the infant remains hypoxic even when given oxygen",
        "CO2": "carbon dioxide level in the infant's blood",
        "ChestXray": "the chest X-ray result",
        "Grunting": "the infant shows grunting, an abnormal breathing sound",
        "LVHreport": "the left ventricular hypertrophy report",
        "LowerBodyO2": "oxygen saturation measured in the lower body",
        "RUQO2": "oxygen saturation measured in the right upper quadrant",
        "CO2Report": "the reported carbon dioxide level",
        "XrayReport": "the reported chest X-ray result",
        "Disease": "the underlying lung disease",
        "GruntingReport": "the reported presence of grunting",
        "Age": "age of the infant in days",
        "LVH": "left ventricular hypertrophy, thickening of the left heart wall",
        "DuctFlow": "ductal flow, blood flow through the ductus arteriosus",
        "CardiacMixing": "mixing of blood between the systemic and pulmonary circulations",
        "LungParench": "the state of the lung parenchyma, the functional lung tissue",
        "LungFlow": "blood flow through the lung",
        "Sick": "whether the infant is considered sick",
    },
    "INSURANCE": {
        "GoodStudent": "the driver is a good student based on school grades",
        "Age": "the age of the driver",
        "SocioEcon": "the socio-economic status of the driver",
        "RiskAversion": "how risk averse the driver is",
        "VehicleYear": "the model year of the vehicle",
        "ThisCarDam": "the damage caused to this car in an accident",
        "RuggedAuto": "how rugged the vehicle is",
        "Accident": "whether a traffic accident occurred",
        "MakeModel": "the make and model of the car",
        "DrivQuality": "the quality of driving",
        "Mileage": "the number of miles driven per year",
        "Antilock": "whether the car has an antilock braking system",
        "DrivingSkill": "the driving skill of the driver",
        "SeniorTrain": "whether the driver took a senior driver training course",
        "ThisCarCost": "the repair cost for this car",
        "Theft": "whether the car was stolen",
        "CarValue": "the value of the car",
        "HomeBase": "the neighborhood where the car is normally parked",
        "AntiTheft": "whether the car has an anti-theft system",
        "PropCost": "the total property cost of the claim",
        "OtherCarCost": "the repair cost for the other car involved",
        "OtherCar": "whether another car was involved",
        "MedCost": "the medical cost of the accident",
        "Cushioning": "the cushioning quality of the car body",
        "Airbag": "whether the car has an airbag",
        "ILiCost": "the cost of injuries and litigation",
        "DrivHist": "the driving history of the driver",
    },
    "HAILFINDER": {
        "N0_7muVerMo": "0-7 km mean vertical motion of the atmosphere",
        "SubjVertMo": "subjective assessment of vertical motion",
        "QGVertMotion": "quasi-geostrophic vertical motion",
        "CombVerMo": "combined vertical motion",
        "AreaMeso_ALS": "area of the mesoscale alpha region relevant to the scenario",
        "SatContMoist": "satellite-derived continuous moisture field",
        "RaoContMoist": "Rao continuous moisture field",
        "CombMoisture": "combined moisture estimate",
        "AreaMoDryAir": "area of mesoscale dry air",
        "VISCloudCov": "cloud cover estimated from visible satellite imagery",
        "IRCloudCover": "cloud cover estimated from infrared satellite imagery",
        "CombClouds": "combined cloud cover estimate",
        "CldShadeOth": "other cloud shading and cloud related features",
        "AMInstabMt": "morning instability over the mountains",
        "InsInMt": "instability in the mountains",
        "WndHodograph": "features of the wind hodograph, the vertical wind profile",
        "OutflowFrMt": "outflow from the mountains",
        "MorningBound": "the morning boundary position",
        "Boundaries": "boundaries such as fronts and outflow boundaries",
        "CldShadeConv": "cloud shading indicating convection",
        "CompPlFcst": "computer model forecast for the plains",
        "CapChange": "change in the strength of the capping inversion",
        "LoLevMoistAd": "low level moisture advection",
        "InsChange": "change in instability",
        "MountainFcst": "thunderstorm forecast for the mountains",
        "Date": "the calendar date of the forecast case",
        "Scenario": "the preconvective scenario type",
        "ScenRelAMCIN": "whether the scenario is relevant to morning convective inhibition",
        "MorningCIN": "morning convective inhibition",
        "AMCINInScen": "morning convective inhibition within the scenario",
        "CapInScen": "the cap strength within the scenario",
        "ScenRelAMIns": "whether the scenario is relevant to morning instability",
        "LIfr12ZDENSd": "lifted index from the 12Z Denver sounding",
        "AMDewptCalPl": "morning dew point on the Colorado plains",
        "AMInsWliScen": "morning instability within the scenario",
        "InsSclInScen": "instability scaled within the scenario",
        "ScenRel3_4": "scenario relevance indicator 3-4",
        "LatestCIN": "the latest convective inhibition estimate",
        "LLIW": "low level instability and wind index",
        "CurPropConv": "current propensity for convection",
        "ScnRelPlFcst": "whether the scenario is relevant to the plains forecast",
        "PlainsFcst": "thunderstorm forecast for the plains",
        "N34StarFcst": "N34 star forecast feature",
        "R5Fcst": "R5 forecast feature",
        "Dewpoints": "dew point values",
        "LowLLapse": "low level lapse rate",
        "MeanRH": "mean relative humidity",
        "MidLLapse": "mid level lapse rate",
        "MvmtFeatures": "movement features of the convective environment",
        "RHRatio": "relative humidity ratio",
        "SfcWndShfDis": "surface wind shift and discontinuity",
        "SynForcng": "synoptic scale forcing",
        "TempDis": "temperature discontinuity",
        "WindAloft": "winds aloft",
        "WindFieldMt": "wind field over the mountains",
        "WindFieldPln": "wind field over the plains",
    },
}


NETWORKS = tuple(VARIABLE_MEANINGS)


# ============================================================
# 变量名（从数据文件表头读取，保证与实验数据一致）
# ============================================================

def load_variables(network: str) -> list[str]:
    """从 data/<NET>/<NET>_n100.txt 的表头读取变量名。"""

    path = DATA_ROOT / network / f"{network}_n100.txt"

    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip()

    return header.split("\t")


def anonymous_names(network: str) -> dict[str, str]:
    """
    生成匿名变量名。

    按字典序排序后映射为 V01, V02, ...，保证同一网络下映射稳定可复现，
    同时避免字母顺序泄漏任何结构信息。
    """

    variables = sorted(load_variables(network))
    return {name: f"V{index:02d}" for index, name in enumerate(variables, start=1)}


def variable_meaning(network: str, variable: str) -> str:
    """取变量含义；缺失时返回通用描述而不是抛错。"""

    return VARIABLE_MEANINGS.get(network, {}).get(
        variable,
        "a variable of this network; no further description is available",
    )


def coverage_report() -> str:
    """打印每个网络的含义覆盖率，用于自检。"""

    lines = []
    for network in NETWORKS:
        variables = load_variables(network)
        known = sum(1 for v in variables if v in VARIABLE_MEANINGS[network])
        lines.append(f"{network}: {known}/{len(variables)} variables described")
    return "\n".join(lines)


if __name__ == "__main__":
    print(coverage_report())
