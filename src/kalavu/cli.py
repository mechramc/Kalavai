"""KALAVU CLI — The Decentralized LLM Training Protocol."""

from pathlib import Path

import click

from kalavu import __version__


@click.group()
@click.version_option(version=__version__, prog_name="kalavu")
def main():
    """KALAVU — 20 GPUs. 1 model. No one trains alone."""


@main.group()
def coop():
    """Manage cooperatives."""


@coop.command()
@click.option("--name", required=True, help="Cooperative name")
@click.option("--modules", default=20, help="Number of module slots")
@click.option("--target-params", default="1B", help="Target parameters per module")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: ./<name>)",
)
@click.option(
    "--corpus",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to text corpus for tokenizer training",
)
@click.option("--vocab-size", default=4096, help="Tokenizer vocabulary size")
@click.option("--seed", default=42, help="Random seed for reproducibility")
def create(
    name: str,
    modules: int,
    target_params: str,
    output_dir: Path | None,
    corpus: Path | None,
    vocab_size: int,
    seed: int,
):
    """Create a new cooperative."""
    from kalavu.coop.create import create_cooperative

    create_cooperative(
        name=name,
        modules=modules,
        target_params=target_params,
        output_dir=output_dir,
        corpus_path=corpus,
        vocab_size=vocab_size,
        seed=seed,
    )


@coop.command()
@click.argument("cooperative")
@click.option("--claim-module", type=int, required=True, help="Module slot to claim")
def join(cooperative: str, claim_module: int):
    """Join a cooperative and claim a module slot."""
    click.echo(f"Joining '{cooperative}', claiming module {claim_module}...")


@coop.command()
@click.argument("cooperative")
def status(cooperative: str):
    """Show cooperative status and module alignment."""
    click.echo(f"Status for cooperative '{cooperative}'...")


@coop.command()
@click.argument("cooperative")
def publish(cooperative: str):
    """Publish fused model to Hugging Face Hub."""
    click.echo(f"Publishing '{cooperative}' to Hugging Face Hub...")


@main.group()
def train():
    """Train and submit modules."""


@train.command()
@click.option("--module", type=int, required=True, help="Module slot number")
def start(module: int):
    """Start training a module."""
    click.echo(f"Starting training for module {module}...")


@train.command()
@click.option("--module", type=int, required=True, help="Module slot number")
def submit(module: int):
    """Submit a trained module checkpoint."""
    click.echo(f"Submitting module {module}...")


@main.group()
def check():
    """Alignment monitoring."""


@check.command()
def post():
    """Post alignment checkpoint to cooperative."""
    click.echo("Posting alignment checkpoint...")


@main.group()
def fuse():
    """Fusion pipeline."""


@fuse.command()
@click.argument("cooperative")
def cluster(cooperative: str):
    """Cluster modules by CKA similarity."""
    click.echo(f"Clustering modules for '{cooperative}'...")


@fuse.command("build")
@click.argument("cooperative")
def build_cmd(cooperative: str):
    """Assemble fusion architecture."""
    click.echo(f"Building fusion architecture for '{cooperative}'...")


@fuse.command()
@click.argument("cooperative")
def train_fuse(cooperative: str):
    """Run post-training curriculum on fused model."""
    click.echo(f"Running post-training for '{cooperative}'...")


if __name__ == "__main__":
    main()
