"""PII検出・マスキングのユニットテスト。

メールアドレス、電話番号、マイナンバー、クレジットカード番号、
住所 (都道府県+市区町村) の検出・マスキングを検証する。
"""

from __future__ import annotations

from src.security.pii_detector import PIIDetectionResult, PIIDetector, PIIMatch, PIIType

# --- PIIType ---


class TestPIIType:
    """PIIType enum のテスト。"""

    def test_has_expected_members(self) -> None:
        assert set(PIIType) == {
            PIIType.EMAIL,
            PIIType.PHONE,
            PIIType.MY_NUMBER,
            PIIType.CREDIT_CARD,
            PIIType.ADDRESS,
        }

    def test_values_are_strings(self) -> None:
        assert PIIType.EMAIL.value == "email"
        assert PIIType.PHONE.value == "phone"
        assert PIIType.MY_NUMBER.value == "my_number"
        assert PIIType.CREDIT_CARD.value == "credit_card"
        assert PIIType.ADDRESS.value == "address"


# --- PIIMatch ---


class TestPIIMatch:
    """PIIMatch モデルのテスト。"""

    def test_creation(self) -> None:
        match = PIIMatch(pii_type=PIIType.EMAIL, start=0, end=15)
        assert match.pii_type == PIIType.EMAIL
        assert match.start == 0
        assert match.end == 15

    def test_no_matched_text_stored(self) -> None:
        """PIIMatch にはマッチしたテキスト自体を保持しない。"""
        fields = set(PIIMatch.model_fields.keys())
        assert "text" not in fields
        assert "value" not in fields
        assert "matched_text" not in fields


# --- PIIDetectionResult ---


class TestPIIDetectionResult:
    """PIIDetectionResult モデルのテスト。"""

    def test_result_without_pii(self) -> None:
        result = PIIDetectionResult(has_pii=False, matches=[], masked_text="hello")
        assert result.has_pii is False
        assert result.matches == []
        assert result.masked_text == "hello"

    def test_result_with_pii(self) -> None:
        matches = [PIIMatch(pii_type=PIIType.EMAIL, start=0, end=15)]
        result = PIIDetectionResult(
            has_pii=True,
            matches=matches,
            masked_text="[EMAIL]",
        )
        assert result.has_pii is True
        assert len(result.matches) == 1


# --- Email Detection ---


class TestEmailDetection:
    """メールアドレス検出のテスト。"""

    def test_standard_email(self) -> None:
        detector = PIIDetector()
        result = detector.detect("連絡先: user@example.com です")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.EMAIL for m in result.matches)
        assert "[EMAIL]" in result.masked_text
        assert "user@example.com" not in result.masked_text

    def test_email_with_dots_and_plus(self) -> None:
        detector = PIIDetector()
        result = detector.detect("first.last+tag@sub.example.co.jp")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.EMAIL for m in result.matches)

    def test_no_false_positive_for_at_sign(self) -> None:
        detector = PIIDetector()
        result = detector.detect("@ は記号です")
        assert not any(m.pii_type == PIIType.EMAIL for m in result.matches)


# --- Phone Detection ---


class TestPhoneDetection:
    """電話番号検出のテスト。"""

    def test_mobile_with_hyphens(self) -> None:
        detector = PIIDetector()
        result = detector.detect("電話: 090-1234-5678")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.PHONE for m in result.matches)
        assert "[PHONE]" in result.masked_text

    def test_mobile_without_hyphens(self) -> None:
        detector = PIIDetector()
        result = detector.detect("09012345678に連絡")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.PHONE for m in result.matches)

    def test_landline_with_hyphens(self) -> None:
        detector = PIIDetector()
        result = detector.detect("TEL: 03-1234-5678")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.PHONE for m in result.matches)

    def test_international_format(self) -> None:
        detector = PIIDetector()
        result = detector.detect("+81-90-1234-5678")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.PHONE for m in result.matches)

    def test_no_false_positive_for_short_number(self) -> None:
        """4桁以下の数字は電話番号と判定しない。"""
        detector = PIIDetector()
        result = detector.detect("部屋番号は1234です")
        assert not any(m.pii_type == PIIType.PHONE for m in result.matches)

    def test_phone_does_not_match_uuid_suffix(self) -> None:
        """UUID末尾の数字列を電話番号と誤検出しないこと。"""
        detector = PIIDetector()
        result = detector.detect("id: b11a415b-2e7a-4748-be90-347d40456732")
        assert result.has_pii is False


# --- My Number Detection ---


class TestMyNumberDetection:
    """マイナンバー (12桁) 検出のテスト。"""

    def test_my_number_with_hyphens(self) -> None:
        detector = PIIDetector()
        result = detector.detect("マイナンバー: 1234-5678-9012")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.MY_NUMBER for m in result.matches)
        assert "[MY_NUMBER]" in result.masked_text

    def test_my_number_with_spaces(self) -> None:
        detector = PIIDetector()
        result = detector.detect("番号 1234 5678 9012 です")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.MY_NUMBER for m in result.matches)

    def test_my_number_continuous(self) -> None:
        detector = PIIDetector()
        result = detector.detect("123456789012")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.MY_NUMBER for m in result.matches)

    def test_not_11_digits(self) -> None:
        """11桁はマイナンバーとして検出しない。"""
        detector = PIIDetector()
        result = detector.detect("12345678901")
        assert not any(m.pii_type == PIIType.MY_NUMBER for m in result.matches)

    def test_not_13_digits(self) -> None:
        """13桁はマイナンバーとして検出しない。"""
        detector = PIIDetector()
        result = detector.detect("1234567890123")
        assert not any(m.pii_type == PIIType.MY_NUMBER for m in result.matches)


# --- Credit Card Detection ---


class TestCreditCardDetection:
    """クレジットカード番号 (16桁) 検出のテスト。"""

    def test_credit_card_with_hyphens(self) -> None:
        detector = PIIDetector()
        result = detector.detect("カード: 1234-5678-9012-3456")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.CREDIT_CARD for m in result.matches)
        assert "[CREDIT_CARD]" in result.masked_text

    def test_credit_card_with_spaces(self) -> None:
        detector = PIIDetector()
        result = detector.detect("1234 5678 9012 3456")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.CREDIT_CARD for m in result.matches)

    def test_credit_card_continuous(self) -> None:
        detector = PIIDetector()
        result = detector.detect("1234567890123456")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.CREDIT_CARD for m in result.matches)

    def test_not_15_digits(self) -> None:
        """15桁はクレジットカードとして検出しない。"""
        detector = PIIDetector()
        result = detector.detect("123456789012345")
        assert not any(m.pii_type == PIIType.CREDIT_CARD for m in result.matches)

    def test_credit_card_not_detected_as_my_number(self) -> None:
        """16桁のクレジットカード番号はマイナンバーとして二重検出しない。"""
        detector = PIIDetector()
        result = detector.detect("1234-5678-9012-3456")
        my_number_matches = [m for m in result.matches if m.pii_type == PIIType.MY_NUMBER]
        assert len(my_number_matches) == 0


# --- Address Detection ---


class TestAddressDetection:
    """住所検出のテスト。"""

    def test_tokyo_address(self) -> None:
        detector = PIIDetector()
        result = detector.detect("住所: 東京都千代田区丸の内1-1-1")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.ADDRESS for m in result.matches)
        assert "[ADDRESS]" in result.masked_text

    def test_osaka_address(self) -> None:
        detector = PIIDetector()
        result = detector.detect("大阪府大阪市北区梅田")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.ADDRESS for m in result.matches)

    def test_prefecture_address(self) -> None:
        detector = PIIDetector()
        result = detector.detect("神奈川県横浜市中区")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.ADDRESS for m in result.matches)

    def test_hokkaido_address(self) -> None:
        detector = PIIDetector()
        result = detector.detect("北海道札幌市中央区")
        assert result.has_pii is True
        assert any(m.pii_type == PIIType.ADDRESS for m in result.matches)

    def test_no_false_positive_for_prefecture_only(self) -> None:
        """都道府県名のみでは住所として検出しない。"""
        detector = PIIDetector()
        result = detector.detect("東京都の天気は晴れです")
        assert not any(m.pii_type == PIIType.ADDRESS for m in result.matches)


# --- Mixed PII Detection ---


class TestMixedPIIDetection:
    """複数種類のPII検出テスト。"""

    def test_multiple_types(self) -> None:
        detector = PIIDetector()
        text = "連絡先: user@example.com, TEL: 090-1234-5678"
        result = detector.detect(text)
        assert result.has_pii is True
        pii_types = {m.pii_type for m in result.matches}
        assert PIIType.EMAIL in pii_types
        assert PIIType.PHONE in pii_types

    def test_multiple_occurrences_of_same_type(self) -> None:
        detector = PIIDetector()
        text = "a@b.com と c@d.com"
        result = detector.detect(text)
        email_matches = [m for m in result.matches if m.pii_type == PIIType.EMAIL]
        assert len(email_matches) == 2


# --- Edge Cases ---


class TestEdgeCases:
    """エッジケースのテスト。"""

    def test_empty_string(self) -> None:
        detector = PIIDetector()
        result = detector.detect("")
        assert result.has_pii is False
        assert result.matches == []
        assert result.masked_text == ""

    def test_no_pii(self) -> None:
        detector = PIIDetector()
        result = detector.detect("これはPIIを含まないテキストです。")
        assert result.has_pii is False
        assert result.masked_text == "これはPIIを含まないテキストです。"

    def test_whitespace_only(self) -> None:
        detector = PIIDetector()
        result = detector.detect("   \n\t  ")
        assert result.has_pii is False


# --- Configurable PII Types ---


class TestConfigurablePIITypes:
    """有効なPII種別の制御テスト。"""

    def test_default_detects_all_types(self) -> None:
        detector = PIIDetector()
        text = "user@example.com 090-1234-5678"
        result = detector.detect(text)
        pii_types = {m.pii_type for m in result.matches}
        assert PIIType.EMAIL in pii_types
        assert PIIType.PHONE in pii_types

    def test_restricted_to_email_only(self) -> None:
        detector = PIIDetector(enabled_types={PIIType.EMAIL})
        text = "user@example.com 090-1234-5678"
        result = detector.detect(text)
        pii_types = {m.pii_type for m in result.matches}
        assert PIIType.EMAIL in pii_types
        assert PIIType.PHONE not in pii_types

    def test_empty_set_detects_nothing(self) -> None:
        detector = PIIDetector(enabled_types=set())
        text = "user@example.com 090-1234-5678"
        result = detector.detect(text)
        assert result.has_pii is False
        assert result.matches == []


# --- Mask Method ---


class TestMaskMethod:
    """mask() メソッドのテスト。"""

    def test_returns_string(self) -> None:
        detector = PIIDetector()
        result = detector.mask("user@example.com")
        assert isinstance(result, str)

    def test_matches_detect_masked_text(self) -> None:
        detector = PIIDetector()
        text = "連絡先: user@example.com"
        assert detector.mask(text) == detector.detect(text).masked_text
