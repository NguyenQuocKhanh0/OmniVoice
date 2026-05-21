import bisect
import unicodedata
from functools import lru_cache
from typing import Optional


# =========================
# Settings
# =========================

ENGLISH_MARKER_CHARS = {"`"}

# Một từ tiếng Việt thường tương ứng gần với một âm tiết.
# Có thể tune theo data thật của bạn.
VIETNAMESE_WORD_WEIGHT = 3.0

# Nếu ref_duration đang dùng đơn vị "giây", nên để None hoặc số nhỏ như 0.3 / 0.5.
# Nếu pipeline của bạn dùng frame và trước đây cần threshold=50 thì đổi lại thành 50.
LOW_DURATION_THRESHOLD = 100

BOOST_STRENGTH = 1.3

COUNT_SPACE = True

# Chỉ áp dụng cộng dồn cho target_text.
# ref_text vẫn tính dấu câu theo base weight, không tăng theo số lần lặp.
PUNCTUATION_REPEAT_BOOST = 0.1


class RuleDurationEstimator:
    """
    Rule-based multilingual duration estimator.

    Logic chính:
    - English words được đánh dấu bằng ký tự marker, ví dụ: `hello
      Marker không được tính vào thời lượng.
    - Vietnamese Latin words không có marker được tính theo số từ.
      Ví dụ: "xin chào" = 2 Vietnamese word units.
    - Non-Latin scripts fallback về rule theo Unicode block.
    - Punctuation vẫn được tính như pause.
    - PUNCTUATION_REPEAT_BOOST chỉ áp dụng cho target_text trong estimate_duration().
      ref_text không cộng dồn punctuation repeat boost.
    """

    def __init__(
        self,
        vietnamese_word_weight: float = VIETNAMESE_WORD_WEIGHT,
        english_marker_chars=None,
        count_space: bool = COUNT_SPACE,
        punctuation_repeat_boost: float = PUNCTUATION_REPEAT_BOOST,
    ):
        self.vietnamese_word_weight = vietnamese_word_weight
        self.english_marker_chars = set(
            english_marker_chars or ENGLISH_MARKER_CHARS
        )
        self.count_space = count_space
        self.punctuation_repeat_boost = punctuation_repeat_boost

        self.weights = {
            # Logographic
            "cjk": 3.0,

            # Syllabic / blocks
            "hangul": 2.5,
            "kana": 2.2,
            "ethiopic": 3.0,
            "yi": 3.0,

            # Abugida
            "indic": 1.8,
            "thai_lao": 1.5,
            "khmer_myanmar": 1.8,

            # Abjad
            "arabic": 1.5,
            "hebrew": 1.5,

            # Alphabet
            "latin": 1.0,
            "cyrillic": 1.0,
            "greek": 1.0,
            "armenian": 1.0,
            "georgian": 1.0,

            # Misc
            "space": 0.2,
            "digit": 3.5,
            "mark": 0.0,
            "default": 1.0,
        }

        self.default_punctuation_weight = 0.5

        # Base weight cho từng dấu câu.
        self.punctuation_weights = {
            ",": 3.5,
            "،": 0.35,
            ";": 4.5,
            ":": 3.5,
            ".": 4.5,
            "。": 4.5,
            "?": 4.5,
            "？": 4.5,
            "!": 4.5,
            "！": 4.5,
            "…": 4.5,
        }

        self.ranges = [
            (0x02AF, "latin"),
            (0x03FF, "greek"),
            (0x052F, "cyrillic"),
            (0x058F, "armenian"),
            (0x05FF, "hebrew"),
            (0x077F, "arabic"),
            (0x089F, "arabic"),
            (0x08FF, "arabic"),
            (0x097F, "indic"),
            (0x09FF, "indic"),
            (0x0A7F, "indic"),
            (0x0AFF, "indic"),
            (0x0B7F, "indic"),
            (0x0BFF, "indic"),
            (0x0C7F, "indic"),
            (0x0CFF, "indic"),
            (0x0D7F, "indic"),
            (0x0DFF, "indic"),
            (0x0EFF, "thai_lao"),
            (0x0FFF, "indic"),
            (0x109F, "khmer_myanmar"),
            (0x10FF, "georgian"),
            (0x11FF, "hangul"),
            (0x137F, "ethiopic"),
            (0x139F, "ethiopic"),
            (0x13FF, "default"),
            (0x167F, "default"),
            (0x169F, "default"),
            (0x16FF, "default"),
            (0x171F, "default"),
            (0x173F, "default"),
            (0x175F, "default"),
            (0x177F, "default"),
            (0x17FF, "khmer_myanmar"),
            (0x18AF, "default"),
            (0x18FF, "default"),
            (0x194F, "indic"),
            (0x19DF, "indic"),
            (0x19FF, "khmer_myanmar"),
            (0x1A1F, "indic"),
            (0x1AAF, "indic"),
            (0x1B7F, "indic"),
            (0x1BBF, "indic"),
            (0x1BFF, "indic"),
            (0x1C4F, "indic"),
            (0x1C7F, "indic"),
            (0x1C8F, "cyrillic"),
            (0x1CBF, "georgian"),
            (0x1CCF, "indic"),
            (0x1CFF, "indic"),
            (0x1D7F, "latin"),
            (0x1DBF, "latin"),
            (0x1DFF, "default"),
            (0x1EFF, "latin"),
            (0x309F, "kana"),
            (0x30FF, "kana"),
            (0x312F, "cjk"),
            (0x318F, "hangul"),
            (0x9FFF, "cjk"),
            (0xA4CF, "yi"),
            (0xA4FF, "default"),
            (0xA63F, "default"),
            (0xA69F, "cyrillic"),
            (0xA6FF, "default"),
            (0xA7FF, "latin"),
            (0xA82F, "indic"),
            (0xA87F, "default"),
            (0xA8DF, "indic"),
            (0xA8FF, "indic"),
            (0xA92F, "indic"),
            (0xA95F, "indic"),
            (0xA97F, "hangul"),
            (0xA9DF, "indic"),
            (0xA9FF, "khmer_myanmar"),
            (0xAA5F, "indic"),
            (0xAA7F, "khmer_myanmar"),
            (0xAADF, "indic"),
            (0xAAFF, "indic"),
            (0xAB2F, "ethiopic"),
            (0xAB6F, "latin"),
            (0xABBF, "default"),
            (0xABFF, "indic"),
            (0xD7AF, "hangul"),
            (0xFAFF, "cjk"),
            (0xFDFF, "arabic"),
            (0xFE6F, "default"),
            (0xFEFF, "arabic"),
            (0xFFEF, "latin"),
        ]

        self.breakpoints = [item[0] for item in self.ranges]

    def _unicode_category(self, char: str) -> str:
        return unicodedata.category(char)

    def _is_word_char(self, char: str) -> bool:
        category = self._unicode_category(char)
        return category.startswith("L") or category.startswith("M")

    def _is_punctuation(self, char: str) -> bool:
        category = self._unicode_category(char)
        return category.startswith("P")

    def _consume_word(self, text: str, start: int) -> tuple[str, int]:
        i = start

        while i < len(text) and self._is_word_char(text[i]):
            i += 1

        return text[start:i], i

    def _script_type_from_code(self, code: int) -> str:
        idx = bisect.bisect_left(self.breakpoints, code)

        if idx < len(self.ranges):
            return self.ranges[idx][1]

        if code > 0x20000:
            return "cjk"

        return "default"

    def _is_latin_word(self, word: str) -> bool:
        """
        True cho cả English và Vietnamese vì đều dùng Latin script.

        Việc phân biệt English/Vietnamese dựa vào marker:
        - Có marker phía trước: English
        - Không có marker: Vietnamese
        """
        has_letter = False

        for char in word:
            category = self._unicode_category(char)

            if category.startswith("M"):
                continue

            if not category.startswith("L"):
                return False

            script_type = self._script_type_from_code(ord(char))

            if script_type != "latin":
                return False

            has_letter = True

        return has_letter

    def _get_punctuation_weight(self, char: str) -> float:
        """
        Lấy base weight của dấu câu.
        """
        return self.punctuation_weights.get(
            char,
            self.default_punctuation_weight,
        )

    def _get_repeated_punctuation_weight(
        self,
        char: str,
        previous_count: int,
        apply_repeat_boost: bool = True,
    ) -> float:
        """
        Tính weight cho dấu câu.

        apply_repeat_boost=True:
            - previous_count = 0 -> lần 1 -> base_weight
            - previous_count = 1 -> lần 2 -> base_weight * (1 + boost)
            - previous_count = 2 -> lần 3 -> base_weight * (1 + boost)^2

        apply_repeat_boost=False:
            - mọi lần đều dùng base_weight, không cộng dồn.
        """
        base_weight = self._get_punctuation_weight(char)

        if not apply_repeat_boost:
            return base_weight

        multiplier = (1.0 + self.punctuation_repeat_boost) ** previous_count
        return base_weight * multiplier

    @lru_cache(maxsize=4096)
    def _get_char_weight(self, char: str) -> float:
        """
        Tính weight cho 1 ký tự.

        Dùng cho:
        - English marked words
        - Các script fallback
        - Space
        - Digit
        - Symbol

        Lưu ý:
        - Dynamic punctuation boost không xử lý ở đây,
          vì hàm này có cache và không biết dấu câu đã xuất hiện bao nhiêu lần.
        - Dynamic punctuation boost được xử lý trong calculate_total_weight().
        """
        if char in self.english_marker_chars:
            return 0.0

        code = ord(char)

        # ASCII Latin fast path
        if (65 <= code <= 90) or (97 <= code <= 122):
            return self.weights["latin"]

        # Whitespace phổ biến
        if char in {" ", "\t", "\n", "\r"}:
            return self.weights["space"] if self.count_space else 0.0

        # Arabic Tatweel: kéo dài hình chữ, không phát âm
        if code == 0x0640:
            return self.weights["mark"]

        category = self._unicode_category(char)

        if category.startswith("M"):
            return self.weights["mark"]

        if category.startswith("Z"):
            return self.weights["space"] if self.count_space else 0.0

        if category.startswith("N"):
            return self.weights["digit"]

        if category.startswith("P"):
            return self._get_punctuation_weight(char)

        if category.startswith("S"):
            return self.weights["default"]

        script_type = self._script_type_from_code(code)

        return self.weights.get(script_type, self.weights["default"])

    def _calculate_marked_english_word_weight(self, word: str) -> float:
        """
        English word sau marker vẫn tính theo ký tự,
        vì độ dài từ tiếng Anh ảnh hưởng tới thời lượng.
        """
        return sum(self._get_char_weight(char) for char in word)

    def _calculate_unmarked_latin_word_weight(self, word: str) -> float:
        """
        Theo convention:
        - English words đều có marker ` phía trước.
        - Latin word không có marker được xem là tiếng Việt.
        """
        return self.vietnamese_word_weight

    def calculate_total_weight(
        self,
        text: str,
        apply_punctuation_repeat_boost: bool = True,
    ) -> float:
        """
        Tính tổng weight của text.

        apply_punctuation_repeat_boost=True:
            Dấu câu lặp lại được boost dần.

        apply_punctuation_repeat_boost=False:
            Dấu câu luôn dùng base weight, không cộng dồn.

        Trong estimate_duration():
            - target_text dùng True
            - ref_text dùng False
        """
        if not text:
            return 0.0

        text = unicodedata.normalize("NFC", text)

        total_weight = 0.0
        punctuation_counts = {}
        i = 0

        while i < len(text):
            char = text[i]

            # English marker: không tính marker.
            # Nếu ngay sau marker là word thì word đó được tính theo English char-based.
            if char in self.english_marker_chars:
                i += 1

                if i < len(text) and self._is_word_char(text[i]):
                    word, i = self._consume_word(text, i)
                    total_weight += self._calculate_marked_english_word_weight(word)

                continue

            # Word không có marker
            if self._is_word_char(char):
                word, i = self._consume_word(text, i)

                if self._is_latin_word(word):
                    total_weight += self._calculate_unmarked_latin_word_weight(word)
                else:
                    total_weight += sum(self._get_char_weight(c) for c in word)

                continue

            # Dấu câu
            if self._is_punctuation(char):
                previous_count = punctuation_counts.get(char, 0)

                total_weight += self._get_repeated_punctuation_weight(
                    char=char,
                    previous_count=previous_count,
                    apply_repeat_boost=apply_punctuation_repeat_boost,
                )

                punctuation_counts[char] = previous_count + 1
                i += 1

                continue

            # Non-word char khác: space, digit, symbol...
            total_weight += self._get_char_weight(char)
            i += 1

        return total_weight

    def estimate_duration(
        self,
        target_text: str,
        ref_text: str,
        ref_duration: float,
        low_threshold: Optional[float] = LOW_DURATION_THRESHOLD,
        boost_strength: float = BOOST_STRENGTH,
    ) -> float:
        """
        Estimate duration dựa trên reference text/audio.

        Quy tắc mới:
            - ref_text: không áp dụng PUNCTUATION_REPEAT_BOOST
            - target_text: có áp dụng PUNCTUATION_REPEAT_BOOST

        Công thức:
            speed = ref_weight / ref_duration
            target_duration = target_weight / speed
        """
        if ref_duration <= 0 or not ref_text:
            return 0.0

        ref_weight = self.calculate_total_weight(
            ref_text,
            apply_punctuation_repeat_boost=False,
        )

        if ref_weight <= 0:
            return 0.0

        target_weight = self.calculate_total_weight(
            target_text,
            apply_punctuation_repeat_boost=True,
        )

        speed_factor = ref_weight / ref_duration
        estimated_duration = target_weight / speed_factor

        # if low_threshold is not None and estimated_duration < low_threshold:
        #     if low_threshold <= 0:
        #         return estimated_duration

        #     alpha = 1.0 / boost_strength

        #     estimated_duration = low_threshold * (
        #         estimated_duration / low_threshold
        #     ) ** alpha
        alpha = 1.0 / boost_strength
        estimated_duration = (
            low_threshold
            * (estimated_duration / (low_threshold * 1.215)) ** alpha
            + 0.151 * estimated_duration**1.155
        )
        return estimated_duration

    def breakdown(
        self,
        text: str,
        apply_punctuation_repeat_boost: bool = True,
    ):
        """
        Debug helper để xem từng token được tính như thế nào.

        apply_punctuation_repeat_boost=True:
            Xem breakdown giống target_text.

        apply_punctuation_repeat_boost=False:
            Xem breakdown giống ref_text.
        """
        if not text:
            return []

        text = unicodedata.normalize("NFC", text)

        rows = []
        punctuation_counts = {}
        i = 0

        while i < len(text):
            char = text[i]

            if char in self.english_marker_chars:
                marker = char
                i += 1

                if i < len(text) and self._is_word_char(text[i]):
                    word, i = self._consume_word(text, i)
                    weight = self._calculate_marked_english_word_weight(word)

                    rows.append(
                        {
                            "token": marker + word,
                            "type": "marked_english_word",
                            "weight": weight,
                        }
                    )
                else:
                    rows.append(
                        {
                            "token": marker,
                            "type": "ignored_marker",
                            "weight": 0.0,
                        }
                    )

                continue

            if self._is_word_char(char):
                word, i = self._consume_word(text, i)

                if self._is_latin_word(word):
                    weight = self._calculate_unmarked_latin_word_weight(word)
                    token_type = "vietnamese_word"
                else:
                    weight = sum(self._get_char_weight(c) for c in word)
                    token_type = "non_latin_word"

                rows.append(
                    {
                        "token": word,
                        "type": token_type,
                        "weight": weight,
                    }
                )

                continue

            if self._is_punctuation(char):
                previous_count = punctuation_counts.get(char, 0)
                occurrence = previous_count + 1

                base_weight = self._get_punctuation_weight(char)

                if apply_punctuation_repeat_boost:
                    multiplier = (
                        1.0 + self.punctuation_repeat_boost
                    ) ** previous_count
                else:
                    multiplier = 1.0

                weight = self._get_repeated_punctuation_weight(
                    char=char,
                    previous_count=previous_count,
                    apply_repeat_boost=apply_punctuation_repeat_boost,
                )

                punctuation_counts[char] = occurrence

                rows.append(
                    {
                        "token": char,
                        "type": "punctuation",
                        "occurrence": occurrence,
                        "base_weight": base_weight,
                        "multiplier": multiplier,
                        "weight": weight,
                    }
                )

                i += 1

                continue

            weight = self._get_char_weight(char)

            if char in {" ", "\t", "\n", "\r"}:
                token_type = "space"
            elif self._unicode_category(char).startswith("N"):
                token_type = "digit"
            elif self._unicode_category(char).startswith("S"):
                token_type = "symbol"
            else:
                token_type = "other"

            rows.append(
                {
                    "token": char,
                    "type": token_type,
                    "weight": weight,
                }
            )

            i += 1

        return rows
