#pragma once

// HOW THE SR CROP READS ITS INPUTS: copied into crop-sized textures of ours, or the game's full-size textures through
// DLSS's subrects. One pure rule, shared so that it is tested where it is written (the static_asserts below fail the
// build).
//
// Why: with the crop reading copies, a crop with no gaze follow shimmered under DL(S)SS, and turning the copies off
// removed most of it. Something
// below us changed the vectors of a crop evaluate that reads COPIES (600 of 600 a window, a
// crop-sized texture of its own) while an evaluate that reads through SUBRECTS came back untouched (0 of 600, the same
// launch). Vectors made for a crop-sized texture cannot be those of the box's place in the eye, and the crop shimmers. The one run in which the
// crop had been judged clean under DL(S)SS had the copies off; they had been turned on later
// to carry a FOLLOWING crop's history over a move (the shift goes into copied vectors), and the shimmer was
// then looked for in the moves.
//
// The rule:
//   - ray reconstruction: always copies. The test switch [Inputs] UpscalerFoveaRrSubrects (Crimson Desert,
//     when the RR crop went through the NGX entry and its copied vectors were replaced below us) was removed:
//     with the crop direct (CD) or through the NGX entry (Outlaws) and 'RR crop: vectors' on DL(S)SS's, the
//     copies came back unswapped (0 / 0 / 0 every window) and CD's NPCs looked fine on 'All copies';
//   - the switch set by hand (the ini's key, or the overlay's 'Crop reads copies'): as set -- the A/B stays;
//   - not set (auto): the default (copies), EXCEPT once something below us was seen to replace copied vectors:
//     subrects. A crop that follows the gaze used to keep the copies (a move was carried over in them
//     alone, and under DL(S)SS it shimmered either way); now a move is carried over through subrects too
//     (PerEyeDlss_Dx12::ShiftComposed: the box of our composed vectors, shifted, in a full-size texture of ours), so
//     following or not makes no difference to the rule.
namespace FoveaCopyRule
{
constexpr bool ReadsCopies(bool rayReconstruction, bool setByHand, bool handValue, bool defaultValue,
                           bool swappedBelow)
{
    if (rayReconstruction)
        return true;
    if (setByHand)
        return handValue;
    if (swappedBelow)
        return false;
    return defaultValue;
}

// RR copies whatever else is true.
static_assert(ReadsCopies(true, true, false, true, true));
static_assert(ReadsCopies(true, false, false, false, true));
// By hand: as set, both ways, whatever is below us.
static_assert(!ReadsCopies(false, true, false, true, false));
static_assert(ReadsCopies(false, true, true, true, true));
// Auto, nothing below us: the default.
static_assert(ReadsCopies(false, false, false, true, false));
// Auto, DL(S)SS below us: subrects, still or following.
static_assert(!ReadsCopies(false, false, false, true, true));
} // namespace FoveaCopyRule
