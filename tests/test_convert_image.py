"""Unit tests for scripts/convert_image.py, focused on palette dithering.

The converter is a standalone script rather than an installed package, so it is
loaded by path the same way tests/test_display_py.py loads the component module.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "scripts" / "convert_image.py"


def _load_converter():
    """Import convert_image.py by path as module ``convert_image``."""
    spec = importlib.util.spec_from_file_location("convert_image", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["convert_image"] = module
    spec.loader.exec_module(module)
    return module


convert_image = _load_converter()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def _solid(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Build a flat RGB byte buffer of a single color."""
    return bytes(rgb) * (width * height)


def _mid_gray(width: int, height: int) -> bytes:
    """Build a flat mid-gray buffer: no palette entry matches, so error diffuses."""
    return _solid(width, height, (128, 128, 128))


def _vertical_split(width: int, height: int) -> bytes:
    """Build an RGB buffer: left half black, right half white, one hard edge."""
    row = bytes(BLACK) * (width // 2) + bytes(WHITE) * (width - width // 2)
    return row * height


def _tonal_split(width: int, height: int) -> bytes:
    """Left half mid-gray, right half dull red.

    Neither color is in the palette, so both halves genuinely accumulate
    quantization error -- unlike a pure black/white split, where every pixel
    maps exactly and no algorithm has anything to diffuse.
    """
    row = bytes((100, 100, 100)) * (width // 2) + bytes((200, 60, 60)) * (
        width - width // 2
    )
    return row * height


def _dither(src: bytes, width: int, height: int, **kwargs) -> bytearray:
    """Run the gradient ditherer and return the resulting palette indices."""
    dst = bytearray(width * height)
    convert_image._dither_gradient(src, dst, width, height, **kwargs)
    return dst


class GradientDitherTests(unittest.TestCase):
    """Behavior of the gradient-based error-diffusion ditherer."""

    def setUp(self) -> None:
        self.flat = convert_image.GRADIENT_FLAT_WEIGHT
        self.edge = convert_image.GRADIENT_EDGE_WEIGHT
        self.grad_max = convert_image.GRADIENT_MAX

    def _defaults(self) -> dict[str, float]:
        return {
            "flat": self.flat,
            "edge": self.edge,
            "grad_max": self.grad_max,
        }

    def test_solid_white_maps_entirely_to_the_white_palette_entry(self) -> None:
        """A flat field has zero gradient and zero error, so nothing bleeds."""
        width, height = 8, 8
        white_index = convert_image._nearest_palette_index(*WHITE)

        result = _dither(
            _solid(width, height, WHITE), width, height, **self._defaults()
        )

        self.assertEqual(set(result), {white_index})

    def test_solid_black_maps_entirely_to_the_black_palette_entry(self) -> None:
        width, height = 8, 8
        black_index = convert_image._nearest_palette_index(*BLACK)

        result = _dither(
            _solid(width, height, BLACK), width, height, **self._defaults()
        )

        self.assertEqual(set(result), {black_index})

    def test_output_length_matches_pixel_count(self) -> None:
        width, height = 5, 3

        result = _dither(
            _solid(width, height, WHITE), width, height, **self._defaults()
        )

        self.assertEqual(len(result), width * height)

    def _seam_error(self, indices, width: int, height: int, columns: int) -> float:
        """Mean squared color error just right of the seam in _tonal_split.

        Error diffused across the boundary drags these pixels away from their
        true color, so a lower number means a crisper edge.
        """
        total = count = 0
        for y in range(height):
            for x in range(width // 2, width // 2 + columns):
                pr, pg, pb = convert_image.PALETTE_COLORS[indices[y * width + x]]
                total += (pr - 200) ** 2 + (pg - 60) ** 2 + (pb - 60) ** 2
                count += 1
        return total / count

    def test_seam_stays_truer_to_its_color_than_floyd_steinberg(self) -> None:
        """Damping diffusion at the edge keeps the boundary pixels less contaminated."""
        width, height = 48, 48
        src = _tonal_split(width, height)

        gradient_result = _dither(src, width, height, flat=1.0, edge=0.0, grad_max=32.0)
        fs_result = bytearray(width * height)
        convert_image._dither_floyd_steinberg(src, fs_result, width, height)

        self.assertLess(
            self._seam_error(gradient_result, width, height, 1),
            self._seam_error(fs_result, width, height, 1),
        )

    def test_the_crispness_advantage_is_local_to_the_edge(self) -> None:
        """Only the seam is damped, so the advantage fades away from the boundary."""
        width, height = 48, 48
        src = _tonal_split(width, height)

        gradient_result = _dither(src, width, height, flat=1.0, edge=0.0, grad_max=32.0)
        fs_result = bytearray(width * height)
        convert_image._dither_floyd_steinberg(src, fs_result, width, height)

        at_seam = self._seam_error(fs_result, width, height, 1) - self._seam_error(
            gradient_result, width, height, 1
        )
        far_from_seam = self._seam_error(
            fs_result, width, height, 8
        ) - self._seam_error(gradient_result, width, height, 8)
        self.assertGreater(at_seam, far_from_seam)

    def test_edge_weight_changes_the_result_at_an_edge(self) -> None:
        """The edge weight must be load-bearing: flat alone cannot explain output."""
        width, height = 32, 32
        src = _tonal_split(width, height)

        damped = _dither(src, width, height, flat=1.0, edge=0.0, grad_max=32.0)
        undamped = _dither(src, width, height, flat=1.0, edge=1.0, grad_max=32.0)

        self.assertNotEqual(list(damped), list(undamped))

    def test_gradient_max_changes_how_much_is_treated_as_edge(self) -> None:
        """grad_max must be load-bearing: it decides which pixels count as edges."""
        width, height = 32, 32
        src = _tonal_split(width, height)

        narrow = _dither(src, width, height, flat=1.0, edge=0.0, grad_max=8.0)
        wide = _dither(src, width, height, flat=1.0, edge=0.0, grad_max=4096.0)

        self.assertNotEqual(list(narrow), list(wide))

    def test_a_flat_region_still_diffuses_while_an_edge_is_damped(self) -> None:
        """The whole point: one image, two regions, two different behaviors."""
        width, height = 32, 32
        src = _tonal_split(width, height)

        result = _dither(src, width, height, flat=1.0, edge=0.0, grad_max=32.0)

        # The flat interior of the left half must dither (more than one index);
        # a damped seam would collapse it to a single nearest-color block.
        interior = {
            result[y * width + x] for y in range(4, height - 4) for x in range(2, 12)
        }
        self.assertGreater(len(interior), 1)

    def test_edge_weight_of_zero_discards_all_error_at_a_hard_edge(self) -> None:
        """With edge=0 the seam is pure nearest-color mapping: no diffusion at all."""
        width, height = 16, 16
        src = _vertical_split(width, height)
        black_index = convert_image._nearest_palette_index(*BLACK)
        white_index = convert_image._nearest_palette_index(*WHITE)

        result = _dither(src, width, height, flat=0.0, edge=0.0, grad_max=self.grad_max)

        expected = [
            black_index if x < width // 2 else white_index
            for _ in range(height)
            for x in range(width)
        ]
        self.assertEqual(list(result), expected)

    def test_flat_weight_of_one_reproduces_floyd_steinberg_on_a_flat_gradient_image(
        self,
    ) -> None:
        """With flat==edge==1.0 no damping applies, so it degrades to Floyd-Steinberg."""
        width, height = 24, 24
        # A smooth horizontal ramp exercises real error diffusion.
        src = bytes(
            v
            for y in range(height)
            for x in range(width)
            for v in ((x * 255) // (width - 1),) * 3
        )

        gradient_result = _dither(src, width, height, flat=1.0, edge=1.0, grad_max=1.0)
        fs_result = bytearray(width * height)
        convert_image._dither_floyd_steinberg(src, fs_result, width, height)

        self.assertEqual(list(gradient_result), list(fs_result))

    def test_is_deterministic(self) -> None:
        width, height = 16, 16
        src = _vertical_split(width, height)

        first = _dither(src, width, height, **self._defaults())
        second = _dither(src, width, height, **self._defaults())

        self.assertEqual(list(first), list(second))

    def test_every_index_is_a_valid_palette_entry(self) -> None:
        width, height = 16, 16
        src = _vertical_split(width, height)

        result = _dither(src, width, height, **self._defaults())

        self.assertTrue(all(0 <= i < len(convert_image.PALETTE_COLORS) for i in result))


class GradientFlatRegionTests(unittest.TestCase):
    """Guards against periodic lock-in in smooth areas.

    Damping scales Floyd-Steinberg's compact 4-neighbour kernel, which shortens
    how long diffused error survives. Push the flat-region weight low enough and
    the error buffer stops wandering and settles into a repeating orbit, which
    shows up as a visible weave or plaid across smooth sky. The shipped default
    of 1.0 diffuses the full error in flat regions and does not lock in; these
    tests exist so a future default cannot quietly wander into the bad range.
    """

    PERIODIC_LIMIT = 0.45

    @staticmethod
    def _periodicity(indices, width: int, height: int) -> float:
        """Strongest small-period 2-D tile match. 1.0 is a perfect lattice."""
        best = 0.0
        for py in range(1, 7):
            for px in range(1, 7):
                if px == 1 and py == 1:
                    continue
                total = hit = 0
                for y in range(height - py):
                    base = y * width
                    shifted = (y + py) * width
                    for x in range(width - px):
                        total += 1
                        if indices[base + x] == indices[shifted + x + px]:
                            hit += 1
                if total and hit / total > best:
                    best = hit / total
        return best

    def test_default_weights_do_not_lock_into_a_pattern(self) -> None:
        width, height = 48, 48
        src = _solid(width, height, (70, 110, 180))

        result = _dither(
            src,
            width,
            height,
            flat=convert_image.GRADIENT_FLAT_WEIGHT,
            edge=convert_image.GRADIENT_EDGE_WEIGHT,
            grad_max=convert_image.GRADIENT_MAX,
        )

        self.assertLess(self._periodicity(result, width, height), self.PERIODIC_LIMIT)

    def test_the_metric_detects_a_heavily_damped_flat_weight(self) -> None:
        """Proves the guard above can actually fail; 0.6 is a known-bad weight."""
        width, height = 48, 48
        src = _solid(width, height, (70, 110, 180))

        result = _dither(src, width, height, flat=0.6, edge=0.3, grad_max=64.0)

        self.assertGreater(
            self._periodicity(result, width, height), self.PERIODIC_LIMIT
        )


class GradientOptionPlumbingTests(unittest.TestCase):
    """The three gradient knobs travel from CLI flags through to options."""

    def _options(self, *argv: str) -> convert_image.ConversionOptions:
        args = convert_image.parse_args([str(SCRIPT_PATH), *argv])
        return convert_image.build_options_from_args(args)

    def test_gradient_is_an_accepted_dither_choice(self) -> None:
        options = self._options("--dither", "gradient")

        self.assertEqual(options.dither, "gradient")

    def test_knobs_default_to_the_module_constants(self) -> None:
        options = self._options("--dither", "gradient")

        self.assertEqual(options.grad_flat, convert_image.GRADIENT_FLAT_WEIGHT)
        self.assertEqual(options.grad_edge, convert_image.GRADIENT_EDGE_WEIGHT)
        self.assertEqual(options.grad_max, convert_image.GRADIENT_MAX)

    def test_knobs_are_overridable_from_the_command_line(self) -> None:
        options = self._options(
            "--dither",
            "gradient",
            "--grad-flat",
            "0.8",
            "--grad-edge",
            "0.2",
            "--grad-max",
            "48",
        )

        self.assertEqual(options.grad_flat, 0.8)
        self.assertEqual(options.grad_edge, 0.2)
        self.assertEqual(options.grad_max, 48.0)

    def test_validate_accepts_gradient_with_default_knobs(self) -> None:
        convert_image.validate_options(self._options("--dither", "gradient"))

    def test_validate_rejects_an_edge_weight_above_one(self) -> None:
        options = self._options("--dither", "gradient", "--grad-edge", "1.5")

        with self.assertRaises(SystemExit) as caught:
            convert_image.validate_options(options)

        self.assertIn("--grad-edge", str(caught.exception))

    def test_validate_rejects_a_negative_flat_weight(self) -> None:
        options = self._options("--dither", "gradient", "--grad-flat", "-0.1")

        with self.assertRaises(SystemExit) as caught:
            convert_image.validate_options(options)

        self.assertIn("--grad-flat", str(caught.exception))

    def test_validate_rejects_a_zero_gradient_max(self) -> None:
        options = self._options("--dither", "gradient", "--grad-max", "0")

        with self.assertRaises(SystemExit) as caught:
            convert_image.validate_options(options)

        self.assertIn("--grad-max", str(caught.exception))

    def test_validate_names_gradient_in_the_dither_error_message(self) -> None:
        options = self._options()
        options.dither = "bogus"

        with self.assertRaises(SystemExit) as caught:
            convert_image.validate_options(options)

        self.assertIn("gradient", str(caught.exception))


class GradientFilenameTests(unittest.TestCase):
    """Filename slugs stay short unless a knob is actually customized."""

    def _options(self, *argv: str) -> convert_image.ConversionOptions:
        args = convert_image.parse_args([str(SCRIPT_PATH), *argv])
        return convert_image.build_options_from_args(args)

    def test_dither_abbreviates_to_grad(self) -> None:
        self.assertEqual(convert_image._short_dither("gradient"), "grad")

    def test_default_knobs_add_no_tokens(self) -> None:
        slug = convert_image._build_filename_slug(self._options("--dither", "gradient"))

        self.assertEqual(slug, "custom-cover-grad")

    def test_customized_knobs_appear_as_tokens(self) -> None:
        options = self._options(
            "--dither",
            "gradient",
            "--grad-flat",
            "0.9",
            "--grad-edge",
            "0.2",
            "--grad-max",
            "48",
        )

        slug = convert_image._build_filename_slug(options)

        self.assertEqual(slug, "custom-cover-grad-gf090-ge020-gm48")

    def test_other_dither_modes_never_get_gradient_tokens(self) -> None:
        options = self._options("--dither", "atkinson", "--grad-edge", "0.2")

        slug = convert_image._build_filename_slug(options)

        self.assertEqual(slug, "custom-cover-atkinson")


class QuantizeDispatchTests(unittest.TestCase):
    """_quantize_to_palette routes to the gradient ditherer and honors its knobs."""

    def _options(self, *argv: str) -> convert_image.ConversionOptions:
        args = convert_image.parse_args([str(SCRIPT_PATH), *argv])
        return convert_image.build_options_from_args(args)

    def _quantized_indices(self, options, src: bytes, width: int, height: int):
        from PIL import Image

        image = Image.frombytes("RGB", (width, height), src)
        result = convert_image._quantize_to_palette(image, options, Image)
        return list(result.tobytes())

    def test_gradient_dispatch_matches_calling_the_ditherer_directly(self) -> None:
        width, height = 16, 16
        src = _vertical_split(width, height)
        options = self._options("--dither", "gradient")
        expected = bytearray(width * height)
        convert_image._dither_gradient(
            src,
            expected,
            width,
            height,
            flat=convert_image.GRADIENT_FLAT_WEIGHT,
            edge=convert_image.GRADIENT_EDGE_WEIGHT,
            grad_max=convert_image.GRADIENT_MAX,
        )

        self.assertEqual(
            self._quantized_indices(options, src, width, height), list(expected)
        )

    def test_knobs_reach_the_ditherer_through_dispatch(self) -> None:
        width, height = 16, 16
        src = _vertical_split(width, height)
        options = self._options(
            "--dither", "gradient", "--grad-flat", "0.0", "--grad-edge", "0.0"
        )
        black_index = convert_image._nearest_palette_index(*BLACK)
        white_index = convert_image._nearest_palette_index(*WHITE)

        indices = self._quantized_indices(options, src, width, height)

        expected = [
            black_index if x < width // 2 else white_index
            for _ in range(height)
            for x in range(width)
        ]
        self.assertEqual(indices, expected)

    def test_atkinson_still_dispatches_correctly(self) -> None:
        width, height = 16, 16
        src = _mid_gray(width, height)
        options = self._options("--dither", "atkinson")
        expected = bytearray(width * height)
        convert_image._dither_atkinson(src, expected, width, height)

        self.assertEqual(
            self._quantized_indices(options, src, width, height), list(expected)
        )

    def test_none_still_dispatches_correctly(self) -> None:
        width, height = 16, 16
        src = _mid_gray(width, height)
        options = self._options("--dither", "none")
        expected = bytearray(width * height)
        convert_image._map_without_dither(src, expected)

        self.assertEqual(
            self._quantized_indices(options, src, width, height), list(expected)
        )

    def test_the_dither_modes_disagree_on_mid_gray(self) -> None:
        """Guards the dispatch tests above: mid-gray must actually distinguish them."""
        width, height = 16, 16
        src = _mid_gray(width, height)
        outputs = []
        for mode in ("none", "atkinson", "gradient"):
            options = self._options("--dither", mode)
            outputs.append(tuple(self._quantized_indices(options, src, width, height)))

        self.assertEqual(len(set(outputs)), len(outputs))


class GuidedModeTests(unittest.TestCase):
    """The guided walkthrough only asks about gradient knobs when they apply."""

    def _run_guided(
        self, dither: str, answers: dict[str, str] | None = None
    ) -> tuple[convert_image.ConversionOptions, list[str]]:
        """Drive guided_mode, accepting every default except the given answers.

        Choice prompts render as ``Label (a/b/c) [default]``, so answers are
        matched on the leading label rather than the full prompt text.
        """
        args = convert_image.parse_args([str(SCRIPT_PATH)])
        prompts: list[str] = []
        replies = {"Dither": dither, **(answers or {})}

        def fake_input(prompt: str) -> str:
            # A rejected answer makes the helper re-ask the same question. Fail
            # fast instead of spinning forever on an answer the prompt refuses.
            if prompts.count(prompt) >= 2:
                raise AssertionError(f"guided_mode kept re-asking: {prompt!r}")
            prompts.append(prompt)
            for label, reply in replies.items():
                if prompt.startswith(label):
                    return reply
            return ""

        with unittest.mock.patch("builtins.input", fake_input):
            options = convert_image.guided_mode(args)
        return options, prompts

    @staticmethod
    def _labels(prompts: list[str]) -> list[str]:
        return [p.split(" [")[0].split(" (")[0] for p in prompts]

    def test_gradient_choice_prompts_for_the_three_knobs(self) -> None:
        options, prompts = self._run_guided("gradient")

        labels = self._labels(prompts)
        self.assertEqual(options.dither, "gradient")
        self.assertIn("Gradient flat weight", labels)
        self.assertIn("Gradient edge weight", labels)
        self.assertIn("Gradient max", labels)

    def test_other_dither_choices_skip_the_knob_prompts(self) -> None:
        options, prompts = self._run_guided("atkinson")

        labels = self._labels(prompts)
        self.assertEqual(options.dither, "atkinson")
        self.assertNotIn("Gradient flat weight", labels)
        self.assertNotIn("Gradient edge weight", labels)
        self.assertNotIn("Gradient max", labels)

    def test_gradient_knobs_keep_their_defaults_when_accepted(self) -> None:
        options, _ = self._run_guided("gradient")

        self.assertEqual(options.grad_flat, convert_image.GRADIENT_FLAT_WEIGHT)
        self.assertEqual(options.grad_edge, convert_image.GRADIENT_EDGE_WEIGHT)
        self.assertEqual(options.grad_max, convert_image.GRADIENT_MAX)

    def test_edge_weight_of_zero_is_accepted(self) -> None:
        """Zero is a valid edge weight (fully crisp), so the prompt must take it."""
        options, _ = self._run_guided("gradient", {"Gradient edge weight": "0"})

        self.assertEqual(options.grad_edge, 0.0)

    def test_gradient_is_offered_in_the_dither_choices(self) -> None:
        """An invalid answer re-prompts; 'gradient' must be accepted first time."""
        _, prompts = self._run_guided("gradient")

        dither_prompts = [p for p in prompts if p.startswith("Dither")]
        self.assertEqual(len(dither_prompts), 1)


class SummaryOutputTests(unittest.TestCase):
    """The summary surfaces the gradient knobs only for the gradient ditherer."""

    def _summary(self, *argv: str) -> str:
        args = convert_image.parse_args([str(SCRIPT_PATH), *argv])
        options = convert_image.build_options_from_args(args)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            convert_image.print_summary(options)
        return buffer.getvalue()

    def test_gradient_summary_reports_the_knobs(self) -> None:
        output = self._summary("--dither", "gradient", "--grad-edge", "0.2")

        self.assertIn("dither:     gradient", output)
        self.assertIn("0.2", output)

    def test_atkinson_summary_omits_the_knobs(self) -> None:
        output = self._summary("--dither", "atkinson")

        self.assertNotIn("gradient", output.lower())


if __name__ == "__main__":
    unittest.main()
