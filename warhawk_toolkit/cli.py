"""Warhawk Toolkit CLI."""

import sys
from pathlib import Path
from typing import Optional

import click

from . import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """Warhawk Toolkit - Extract and convert Warhawk PS3 game files.

    \b
    Quick start:
        warhawk full game.psarc

    This extracts everything: files, textures (as DDS), and 3D models (as OBJ).
    Output goes to game_extracted/ by default.
    """
    pass


def _do_extract(archive: Path, output: Optional[Path], convert: bool, list_only: bool):
    """Shared extraction logic for extract and full commands."""
    from .psarc import PSARCReader

    click.echo(f"Opening: {archive}")

    with PSARCReader(archive) as reader:
        if list_only:
            click.echo(f"\nFiles in archive ({len(reader.entries) - 1}):")
            for filename in reader.list_files():
                click.echo(f"  {filename}")
            return

        if output is None:
            output = archive.parent / f"{archive.stem}_extracted"

        click.echo(f"Output:  {output}")
        click.echo(f"Convert: {'yes' if convert else 'no'}")
        click.echo()

        # Extract all files
        extracted_count = 0
        converted_count = 0
        ngp_files = []

        with click.progressbar(
            list(reader.extract_all(output)),
            label="Extracting",
            item_show_func=lambda x: x[0] if x else "",
        ) as items:
            for filename, path in items:
                extracted_count += 1

                # Track NGP files for texture extraction
                if path.suffix.lower() == ".ngp":
                    ngp_files.append(path)

                # Auto-convert RTT files
                if convert and path.suffix.lower() == ".rtt":
                    converted_count += auto_convert_rtt(path)

        # Extract textures and models from NGP files (after all files extracted)
        model_count = 0
        if convert and ngp_files:
            click.echo()
            click.echo("Extracting textures from NGP files...")
            for ngp_path in ngp_files:
                count = extract_ngp_textures(ngp_path, convert_to_dds=True)
                if count > 0:
                    click.echo(f"  {ngp_path.name}: {count} textures")
                    converted_count += count

            click.echo()
            click.echo("Extracting 3D models from NGP files...")
            for ngp_path in ngp_files:
                count = extract_ngp_models(ngp_path, export_textures=True)
                if count > 0:
                    click.echo(f"  {ngp_path.name}: {count} models")
                    model_count += count

        click.echo()
        click.echo(f"Extracted: {extracted_count} files")
        if convert:
            click.echo(f"Converted: {converted_count} textures")
            click.echo(f"Models:    {model_count} models")


@main.command()
@click.argument("archive", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output directory (default: <archive_name>_extracted)",
)
def full(archive: Path, output: Optional[Path]):
    """Extract everything from a PSARC archive (recommended).

    This is the easiest way to use Warhawk Toolkit. It does everything:

    \b
    1. Extracts all files from the PSARC archive
    2. Extracts textures from NGP model files
    3. Converts all textures to DDS format
    4. Exports 3D models as OBJ files

    \b
    Example:
        warhawk full game.psarc
    """
    try:
        _do_extract(archive, output, convert=True, list_only=False)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("archive", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output directory (default: <archive_name>_extracted)",
)
@click.option(
    "--convert/--no-convert",
    default=True,
    help="Auto-convert Warhawk formats (extract NGP textures, RTT→DDS)",
)
@click.option(
    "--list-only",
    is_flag=True,
    help="List files without extracting",
)
def extract(archive: Path, output: Optional[Path], convert: bool, list_only: bool):
    """Extract files from a PSARC archive (advanced).

    Same as 'warhawk full' but with more options. Use --no-convert to
    skip texture conversion, or --list-only to preview contents.

    \b
    Options:
    - NGP files → Extract embedded textures from paired VRAM
    - RTT textures → DDS format
    """
    try:
        _do_extract(archive, output, convert, list_only)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("ngp_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--vram",
    type=click.Path(exists=True, path_type=Path),
    help="VRAM texture file (auto-detected if not specified)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output directory",
)
@click.option(
    "--dds/--no-dds",
    default=True,
    help="Convert extracted textures to DDS",
)
def ngp(ngp_file: Path, vram: Optional[Path], output: Optional[Path], dds: bool):
    """Extract textures from an NGP file.

    Extracts embedded textures from the NGP file and its paired VRAM file.
    Textures are saved as RTT files and optionally converted to DDS.

    If --vram is not specified, looks for a .vram file with the same name.
    """
    from .formats import NGPFile

    click.echo(f"Loading: {ngp_file}")

    try:
        ngp_data = NGPFile.from_file(ngp_file, vram)

        click.echo(f"Textures: {ngp_data.texture_count}")
        click.echo()

        if ngp_data.texture_count == 0:
            click.echo("No textures found.")
            return

        if output is None:
            output = ngp_file.parent

        # Extract textures
        for i, rtt_path in ngp_data.extract_textures(output, ngp_file.stem):
            click.echo(f"Extracted: {rtt_path.name}")

            if dds:
                try:
                    from .converters import convert_rtt_to_dds

                    dds_path = rtt_path.with_suffix(".dds")
                    convert_rtt_to_dds(rtt_path, dds_path)
                    click.echo(f"Converted: {dds_path.name}")
                except Exception as e:
                    click.echo(f"  Warning: DDS conversion failed: {e}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("rtt_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output DDS file path",
)
def rtt2dds(rtt_file: Path, output: Optional[Path]):
    """Convert an RTT texture to DDS format.

    DDS is a standard texture format supported by most image viewers
    and 3D applications.
    """
    from .converters import convert_rtt_to_dds
    from .formats import RTTTexture

    click.echo(f"Loading: {rtt_file}")

    try:
        rtt = RTTTexture.from_file(rtt_file)

        click.echo(f"Size:        {rtt.width}x{rtt.height}")
        click.echo(f"Compression: 0x{rtt.compression_type:02X}")
        click.echo(f"Mipmaps:     {rtt.mipmap_count}")
        click.echo()

        if output is None:
            output = rtt_file.with_suffix(".dds")

        convert_rtt_to_dds(rtt, output)
        click.echo(f"Created: {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("ngp_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--vram",
    type=click.Path(exists=True, path_type=Path),
    help="VRAM texture file (auto-detected if not specified)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output directory",
)
@click.option(
    "--textures/--no-textures",
    default=True,
    help="Export textures as DDS",
)
@click.option(
    "--uv2/--uv1",
    default=False,
    help="Use UV2 (chroma/skin) coordinates instead of UV1 (body)",
)
def models(ngp_file: Path, vram: Optional[Path], output: Optional[Path], textures: bool, uv2: bool):
    """Extract 3D models from an NGP file.

    Extracts model geometry as OBJ files with optional MTL materials
    and DDS textures. Supports both Type 1 (Static Mesh) and Type 2
    (Rigged Mesh) models.

    \b
    Features:
    - Vertex normals for proper shading
    - UV1 (body texture) and UV2 (chroma/skin) coordinates
    - Automatic texture extraction

    \b
    UV Coordinate Sets:
    - UV1 (default): For diffuse/body textures
    - UV2 (--uv2):   For chroma/skin textures (team colors, decals)

    If your chroma skin appears misaligned in-game, you likely designed
    it using UV1 coordinates. Re-export with --uv2 and redesign using
    those UV coordinates.

    If --vram is not specified, looks for a .vram file with the same name.
    Use 'warhawk uv-compare' to analyze UV differences before designing skins.
    """
    from .converters import extract_models_from_ngp

    click.echo(f"Loading: {ngp_file}")
    if uv2:
        click.echo("UV mode: UV2 (chroma/skin)")
        click.echo()
        click.echo("NOTE: Using UV2 coordinates for chroma/skin texture mapping.")
        click.echo("      Design your chroma texture using these UV coordinates.")
    else:
        click.echo("UV mode: UV1 (body/diffuse)")
        click.echo()
        click.echo("TIP: For chroma skins, use --uv2 flag instead:")
        click.echo(f"     warhawk models --uv2 {ngp_file.name}")

    click.echo()

    try:
        if output is None:
            output = ngp_file.parent

        model_count = 0
        static_count = 0
        rigged_count = 0
        for obj_path, mtl_path, dds_path in extract_models_from_ngp(
            ngp_file, output, vram, export_textures=textures, use_uv2=uv2
        ):
            model_count += 1
            if "_static" in obj_path.name:
                static_count += 1
            elif "_rigged" in obj_path.name:
                rigged_count += 1
            click.echo(f"Extracted: {obj_path.name}")
            if mtl_path:
                click.echo(f"  Material: {mtl_path.name}")
            if dds_path:
                click.echo(f"  Texture:  {dds_path.name}")

        click.echo()
        click.echo(f"Extracted {model_count} models ({static_count} static, {rigged_count} rigged)")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command("uv-compare")
@click.argument("ngp_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--vram",
    type=click.Path(exists=True, path_type=Path),
    help="VRAM texture file (auto-detected if not specified)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output directory for comparison OBJ files",
)
@click.option(
    "--export-both",
    is_flag=True,
    help="Export separate OBJ files for UV1 and UV2 comparison in Blender",
)
def uv_compare(ngp_file: Path, vram: Optional[Path], output: Optional[Path], export_both: bool):
    """Analyze UV coordinate differences between UV1 and UV2.

    This diagnostic tool helps identify chroma skin misalignment issues
    by comparing the two UV coordinate sets used in Warhawk models:

    \b
    - UV1 (linker 0x00080003): Used for diffuse/body textures
    - UV2 (linker 0x000A0003): Used for chroma/skin textures

    \b
    Common issue: Designing chroma textures using UV1 coordinates in
    Blender, when the game actually applies chroma using UV2 coordinates.

    Use --export-both to generate separate OBJ files for each UV set,
    allowing side-by-side comparison in Blender.
    """
    from .converters.ngp_to_obj import (
        NGPModel,
        analyze_uv_differences,
        extract_model,
        find_models,
        write_obj,
    )

    click.echo(f"Loading: {ngp_file}")
    click.echo()

    try:
        ngp_data = ngp_file.read_bytes()

        vram_data = None
        if vram and Path(vram).exists():
            vram_data = Path(vram).read_bytes()
        else:
            auto_vram = ngp_file.with_suffix(".vram")
            if auto_vram.exists():
                vram_data = auto_vram.read_bytes()
                click.echo(f"Using VRAM: {auto_vram.name}")

        if output is None:
            output = ngp_file.parent
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

        click.echo("=" * 60)
        click.echo("UV COORDINATE ANALYSIS")
        click.echo("=" * 60)
        click.echo()

        model_count = 0
        for header_offset, header_length, model_type in find_models(ngp_data):
            model = extract_model(ngp_data, vram_data, header_offset, model_type)
            if model is None or not model.vertices:
                continue

            model_count += 1
            type_str = "Static" if model_type == 1 else "Rigged"

            click.echo(f"Model #{model_count}: 0x{header_offset:X} ({type_str})")
            click.echo(f"  Vertices: {len(model.vertices)}")
            click.echo(f"  Faces:    {len(model.faces)}")

            analysis = analyze_uv_differences(model)

            click.echo(f"  UV1 coords: {analysis['uv1_count']}")
            click.echo(f"  UV2 coords: {analysis['uv2_count']}")

            if not analysis['uv1_present']:
                click.echo("  ⚠ UV1 NOT PRESENT - model has no diffuse UV mapping")
            if not analysis['uv2_present']:
                click.echo("  ⚠ UV2 NOT PRESENT - model has no chroma UV mapping")
                click.echo("    (Chroma textures cannot be applied to this model)")

            if analysis['uv1_present'] and analysis['uv2_present']:
                if analysis['identical']:
                    click.echo("  ✓ UV1 and UV2 are IDENTICAL")
                    click.echo("    (Chroma and diffuse share same UV layout)")
                else:
                    click.echo(f"  ✗ UV1 and UV2 are DIFFERENT")
                    click.echo(f"    Overlap:        {analysis['overlap_percentage']:.1f}%")
                    click.echo(f"    Avg difference: {analysis['avg_difference']:.4f}")
                    click.echo(f"    Max difference: {analysis['max_difference']:.4f}")
                    click.echo(f"    Different coords: {analysis['different_coords_count']}")
                    click.echo()
                    click.echo("    ⚠ WARNING: If you designed your chroma texture using UV1")
                    click.echo("      coordinates, it will appear misaligned in-game!")
                    click.echo("      Re-export with: warhawk models --uv2 <file>")

            click.echo()

            # Export comparison OBJ files if requested
            if export_both and model.vertices and model.faces:
                base_name = ngp_file.stem
                model_name = f"{base_name}_0x{header_offset:x}"

                if model.uvs:
                    uv1_path = output / f"{model_name}_UV1.obj"
                    write_obj(model, uv1_path, None, use_uv2=False)
                    click.echo(f"  Exported UV1: {uv1_path.name}")

                if model.uvs2:
                    uv2_path = output / f"{model_name}_UV2.obj"
                    write_obj(model, uv2_path, None, use_uv2=True)
                    click.echo(f"  Exported UV2: {uv2_path.name}")

                if model.uvs and model.uvs2:
                    click.echo("    → Import both into Blender to compare UV layouts")

                click.echo()

        if model_count == 0:
            click.echo("No models found in NGP file.")
        else:
            click.echo("=" * 60)
            click.echo("RECOMMENDATION")
            click.echo("=" * 60)
            click.echo()
            click.echo("For chroma/skin textures, ALWAYS use UV2 coordinates:")
            click.echo()
            click.echo("  warhawk models --uv2 <ngp_file> -o ./output")
            click.echo()
            click.echo("Then design your chroma texture in Blender using the")
            click.echo("UV2 layout from the exported OBJ file.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("loc_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file path",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "text"]),
    default="json",
    help="Output format",
)
def loc(loc_file: Path, output: Optional[Path], output_format: str):
    """Extract localization strings from a LOC file.

    Outputs to JSON, CSV, or plain text format.
    """
    from .formats import LOCFile

    click.echo(f"Loading: {loc_file}")

    try:
        loc_data = LOCFile.from_file(loc_file)

        click.echo(f"Entries: {loc_data.entry_count}")
        click.echo()

        if output is None:
            output = loc_file.with_suffix(f".{output_format}")

        if output_format == "json":
            loc_data.save_json(output)
        elif output_format == "csv":
            loc_data.save_csv(output)
        else:
            loc_data.save_text(output)

        click.echo(f"Created: {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def auto_convert_rtt(path: Path) -> int:
    """Convert an RTT file to DDS. Returns 1 if converted, 0 otherwise."""
    try:
        from .converters import convert_rtt_to_dds

        convert_rtt_to_dds(path, path.with_suffix(".dds"))
        return 1
    except Exception:
        return 0


def extract_ngp_textures(ngp_path: Path, convert_to_dds: bool = True) -> int:
    """Extract textures from an NGP file. Returns count of textures extracted."""
    try:
        from .formats import NGPFile

        ngp = NGPFile.from_file(ngp_path)
        if ngp.texture_count == 0:
            return 0

        count = 0
        for i, rtt_path in ngp.extract_textures(ngp_path.parent, ngp_path.stem):
            count += 1

            if convert_to_dds:
                try:
                    from .converters import convert_rtt_to_dds

                    convert_rtt_to_dds(rtt_path, rtt_path.with_suffix(".dds"))
                except Exception:
                    pass

        return count
    except Exception:
        return 0


def extract_ngp_models(ngp_path: Path, export_textures: bool = True) -> int:
    """Extract 3D models from an NGP file. Returns count of models extracted."""
    try:
        from .converters import extract_models_from_ngp

        count = 0
        for obj_path, mtl_path, dds_path in extract_models_from_ngp(
            ngp_path, ngp_path.parent, export_textures=export_textures
        ):
            count += 1

        return count
    except Exception:
        return 0


if __name__ == "__main__":
    main()
