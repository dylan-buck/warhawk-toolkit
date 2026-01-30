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

    A 3-stage extraction pipeline:

    \b
    Stage 1: PSARC archive extraction
    Stage 2: Warhawk container extraction (NGP + VRAM → RTT textures)
    Stage 3: Format conversion (RTT → DDS)
    """
    pass


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
    """Extract files from a PSARC archive.

    If --convert is enabled (default), Warhawk formats are automatically
    processed:

    \b
    - NGP files → Extract embedded textures from paired VRAM
    - RTT textures → DDS format
    """
    from .psarc import PSARCReader

    click.echo(f"Opening: {archive}")

    try:
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

    If --vram is not specified, looks for a .vram file with the same name.
    Use --uv2 to export with chroma/skin UV coordinates instead of body UVs.
    """
    from .converters import extract_models_from_ngp

    click.echo(f"Loading: {ngp_file}")
    if uv2:
        click.echo("UV mode: UV2 (chroma/skin)")

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
