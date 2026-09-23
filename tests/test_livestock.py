import pandas as pd

from waterpath_data_service.services.livestock import _fao_country_total


def test_fao_country_total_returns_matching_value() -> None:
    fao = pd.DataFrame(
        {
            "Area Code (ISO3)": ["UGA", "UGA", "KEN"],
            "Item": ["Asses", "Asses", "Asses"],
            "Year": [2020, 2019, 2020],
            "Value": [19373, 19243, 900000],
        }
    )

    assert _fao_country_total(fao, "uga", "Asses", 2020) == 19373.0


def test_fao_country_total_distinguishes_zero_from_missing() -> None:
    fao = pd.DataFrame(
        {
            "Area Code (ISO3)": ["UGA"],
            "Item": ["Ducks"],
            "Year": [2020],
            "Value": [0],
        }
    )

    assert _fao_country_total(fao, "UGA", "Ducks", 2020) == 0.0
    assert _fao_country_total(fao, "UGA", "Camels", 2020) is None