"""Portable paths and integrity checks for the archived benchmark."""
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def expected_manifest(name):
    reference = HERE / 'reference_manifests' / (name + '.json')
    if not reference.exists():
        reference = HERE / 'data' / name / 'manifest.json'
    return json.loads(reference.read_text())


def verify_file(path, metadata):
    path = Path(path)
    return (path.is_file() and path.stat().st_size == metadata['bytes']
            and sha256(path) == metadata['sha256'])


def prepared_data_matches(name):
    expected = expected_manifest(name)
    directory = HERE / 'data' / name
    manifest = directory / 'manifest.json'
    if not manifest.exists():
        return False
    try:
        actual = json.loads(manifest.read_text())
    except (ValueError, OSError):
        return False
    fields = ('files', 'data_sha256', 'sizes', 'source_rows_loaded', 'features',
              'feature_names', 'class_count', 'category_mappings',
              'normalisation_statistics', 'dropped_columns')
    return (all(actual.get(key) == expected.get(key) for key in fields)
            and all(verify_file(directory / filename, meta)
                    for filename, meta in expected['files'].items()))



def check_prepared_manifest(name, actual):
    expected = expected_manifest(name)
    for field in ('files', 'data_sha256', 'sizes', 'source_rows_loaded', 'features',
                  'feature_names', 'class_count', 'category_mappings',
                  'normalisation_statistics', 'dropped_columns'):
        if actual[field] != expected[field]:
            raise ValueError(f'{name}: rebuilt {field} differs from the archived experiment; '
                             'refusing to accept a different dataset')


def download_verified(url, destination, metadata):
    """Stream into a temporary file; accept only the recorded bytes and SHA-256."""
    destination = Path(destination)
    if verify_file(destination, metadata):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.part')
    print('DOWNLOAD', url, flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'SUANN-benchmark-reproduction/1'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as out:
            while block := response.read(8 * 1024 * 1024):
                out.write(block)
        if not verify_file(temporary, metadata):
            raise ValueError(f'Source checksum/size mismatch: {url}')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def hub_source(repo, filename):
    prefix = f'https://huggingface.co/datasets/{repo}/resolve/'
    for name in json.loads((HERE / 'protocol.json').read_text())['datasets']:
        for source in expected_manifest(name)['sources']:
            url = source['url']
            if url.startswith(prefix) and url[len(prefix):].partition('/')[2] == filename:
                revision = url[len(prefix):].partition('/')[0]
                if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
                    raise ValueError(f'Unpinned source revision: {url}')
                path = HERE / 'sources' / 'huggingface' / repo / revision / filename
                return download_verified(url, path, source)
    raise ValueError(f'No archived source version for {repo}/{filename}')


def bundled_source(name, filename):
    path = HERE / 'inputs' / filename
    if not verify_file(path, expected_manifest(name)['sources'][0]):
        raise ValueError(f'Missing or modified bundled source: {path}')
    return path


def ensure_directories():
    for name in ('data', 'sources', 'runs', 'logs', 'figures', 'verification', 'checkpoints'):
        (HERE / name).mkdir(parents=True, exist_ok=True)


def require_workspace(root=HERE):
    """Training must use a snapshot initialised by the reproduction launcher."""
    root = Path(root).resolve()
    marker = root / 'REPRODUCTION.json'
    if not marker.is_file():
        raise RuntimeError('Use reproduce.py to start a fresh experiment in this folder, '
                           'or reproduce.py --output NEW_FOLDER for a separate copy')
    record = json.loads(marker.read_text())
    if record.get('format') != 'suann-full-reproduction-v1':
        raise ValueError('Unrecognised reproduction workspace')
    if not record.get('snapshot_sha256'):
        raise ValueError('Reproduction snapshot has no integrity manifest')
    for relative, expected in record['snapshot_sha256'].items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError(f'Invalid snapshot path: {relative}')
        if sha256(path) != expected:
            raise ValueError(f'Reproduction snapshot changed: {relative}; start a fresh run without --resume')
    return record
