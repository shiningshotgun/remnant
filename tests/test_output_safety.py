"""Tests for safe CLI argument parsing and destination publication contract."""

import os
import shutil
import tempfile
import unittest

from remnant_core.study import parse_args, run_reach_study


class TestOutputSafety(unittest.TestCase):
    def test_cli_argument_parsing(self):
        """CLI requires --output-dir / -o and parses it correctly."""
        # Valid --output-dir
        args = parse_args(["--output-dir", "custom_results"])
        self.assertEqual(args.output_dir, "custom_results")

        # Valid -o
        args_short = parse_args(["-o", "custom_results_short"])
        self.assertEqual(args_short.output_dir, "custom_results_short")

        # Missing required argument raises SystemExit
        with self.assertRaises(SystemExit):
            parse_args([])

    def test_destination_already_exists_fails_cleanly(self):
        """Passing an existing directory raises FileExistsError without modifying it."""
        temp_dir = tempfile.mkdtemp()
        try:
            sentinel_file = os.path.join(temp_dir, "sentinel.txt")
            with open(sentinel_file, "w") as fp:
                fp.write("do not touch")

            with self.assertRaises(FileExistsError):
                run_reach_study(seeds=[101], max_loops=1, output_dir=temp_dir)

            # Sentinel file must still be present and intact
            self.assertTrue(os.path.exists(sentinel_file))
            with open(sentinel_file, "r") as fp:
                self.assertEqual(fp.read(), "do not touch")
        finally:
            shutil.rmtree(temp_dir)

    def test_staging_cleanup_on_error(self):
        """If run fails midway, staging directories are removed and target does not exist."""
        temp_parent = tempfile.mkdtemp()
        try:
            target_dir = os.path.join(temp_parent, "new_run_dir")

            with self.assertRaises(ValueError):
                # Pass invalid seed list to cause an error
                run_reach_study(seeds=[False], max_loops=1, output_dir=target_dir)  # type: ignore

            self.assertFalse(os.path.exists(target_dir))
            staging_leftovers = [f for f in os.listdir(temp_parent) if f.startswith(".staging_")]
            self.assertEqual(len(staging_leftovers), 0)
        finally:
            shutil.rmtree(temp_parent)



class TestPublicationFaults(unittest.TestCase):
    def test_bounded_cli_and_invalid_parameters_before_execution(self):
        from unittest.mock import patch
        args = parse_args(['-o', 'new', '--seeds', '101', '--max-loops', '1'])
        self.assertEqual((args.seeds, args.max_loops), ([101], 1))
        for options in (['--max-loops', '0'], ['--max-loops', '101'],
                        ['--seeds', '1', '1'], ['--overwrite']):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                parse_args(['-o', 'new'] + options)
        with tempfile.TemporaryDirectory() as directory:
            for seeds, horizon in (([False], 1), ([], 1), ([1, 1], 1), ([1], False), ([1], 101)):
                with patch('remnant_core.study.run_reach_seed') as run:
                    with self.assertRaises(ValueError):
                        run_reach_study(seeds, horizon, os.path.join(directory, 'new'))
                    run.assert_not_called()
                self.assertEqual(os.listdir(directory), [])

    def test_mid_generation_missing_artifact_and_publish_failures(self):
        from unittest.mock import patch
        import remnant_core.study as study
        original = study.write_study_manifest
        for failure in ('generation', 'missing', 'publish'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as parent:
                target = os.path.join(parent, 'new')
                stale = os.path.join(parent, '.staging_someone_else')
                os.mkdir(stale)
                def broken_manifest(*args, **kwargs):
                    original(*args, **kwargs)
                    os.remove(os.path.join(kwargs['output_dir'], 'learners.json'))
                if failure == 'generation':
                    context = patch.object(study, 'write_study_manifest', side_effect=RuntimeError('injected generation'))
                elif failure == 'missing':
                    context = patch.object(study, 'write_study_manifest', side_effect=broken_manifest)
                else:
                    context = patch.object(study, '_publish_new_directory', side_effect=RuntimeError('injected publication'))
                with context, self.assertRaises(RuntimeError):
                    run_reach_study([101], 1, target)
                self.assertFalse(os.path.lexists(target))
                self.assertEqual(os.listdir(parent), ['.staging_someone_else'])

    def test_destination_race_preserves_even_empty_directory(self):
        from unittest.mock import patch
        import remnant_core.study as study
        exclusive_rename = study._publish_new_directory
        for destination_type in ('empty', 'manifest', 'symlink'):
            with self.subTest(destination_type=destination_type), tempfile.TemporaryDirectory() as parent:
                target = os.path.join(parent, 'new')
                marker = os.path.join(parent, 'marker')
                with open(marker, 'w') as stream:
                    stream.write('untouched')
                def race(source, destination):
                    if destination_type == 'symlink':
                        os.symlink(marker, destination)
                    else:
                        os.mkdir(destination)
                        if destination_type == 'manifest':
                            with open(os.path.join(destination, 'manifest.json'), 'w') as stream:
                                stream.write('prior manifest')
                    self.assertTrue(os.path.lexists(source))
                    return exclusive_rename(source, destination)
                with patch.object(study, '_publish_new_directory', side_effect=race), self.assertRaises(FileExistsError):
                    run_reach_study([101], 1, target)
                self.assertTrue(os.path.lexists(target))
                with open(marker) as stream:
                    self.assertEqual(stream.read(), 'untouched')
                if destination_type == 'manifest':
                    with open(os.path.join(target, 'manifest.json')) as stream:
                        self.assertEqual(stream.read(), 'prior manifest')
                elif destination_type == 'empty':
                    self.assertEqual(os.listdir(target), [])
                else:
                    self.assertTrue(os.path.islink(target))
                self.assertFalse(any(name.startswith('.staging_') for name in os.listdir(parent)))

    def test_dangling_symlink_is_protected(self):
        with tempfile.TemporaryDirectory() as parent:
            target = os.path.join(parent, 'new')
            os.symlink(os.path.join(parent, 'absent'), target)
            with self.assertRaises(FileExistsError):
                run_reach_study([101], 1, target)
            self.assertTrue(os.path.islink(target))


if __name__ == "__main__":
    unittest.main()
