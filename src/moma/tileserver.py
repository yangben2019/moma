"""
Tile server module
"""

"""
 #############################################################################
 #                            _     _     _     _                            #
 #                           / \   / \   / \   / \                           #
 #                          ( M ) ( O ) ( M ) ( A )                          #
 #                           \_/   \_/   \_/   \_/                           #
 #                                                                           #
 #               MOMA: Ground vehicles mobility mapping toolkit              #
 #############################################################################

MIT License

Copyright (c) 2024 Dario Sirangelo, Nikolaj Jes Petersen

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import subprocess


class TileServer:
    """Simple interface to Terracotta tile server"""

    def __init__(self, input_path, output_path, database_path):
        """Tile server object constructor. Defines a workspace with an input raster folder path and output folders for the processed rasters and the tile server database. At the current time it relies striclty on file names to label rasters.

        Args:
            input_path (str): Path to the directory holding the site rasters.
            output_path (str): Path to the directory where processed rasters will be stored.
            database_path (str): Path to the Terracotta SQLite database.
        """

        # Define paths
        self.input_path = input_path
        self.output_path = output_path
        self.database_path = database_path

        # Define file path mask for raster files (select all *.tif)
        self.rasters_mask = input_path + "/*.tif"

        # Define mask to identify rasters in the database . Currently uses the entire file name.
        self.optimized_rasters_mask = output_path + "/{name}.tif"

    def update(self):
        """Update the tile server database with optimised versions of the rasters"""

        # Run raster optimization process
        subprocess.run(
            [
                "terracotta",
                "optimize-rasters",
                "--output-folder",
                f"{self.output_path}",
                "--overwrite",
                f"{self.rasters_mask}",
            ],
            check=True,
        )

        # Run database generation/update process
        subprocess.run(
            [
                "terracotta",
                "ingest",
                "--output-file",
                "terracotta.sqlite",
                f"{self.optimized_rasters_mask}",
            ],
            check=True,
        )

    def serve(self):
        """Run the Terracotta XYZ tile server"""

        # Run Terracotta XYZ tile server process
        subprocess.run(
            [
                "terracotta",
                "serve",
                "--database",
                f"{self.database_path}/aumap.sqlite",
                "--port",
                "5100",
            ],
            check=True,
        )

    def connect(self):
        """Run the Terracotta XYZ tile client"""

        # Run Terracotta XYZ tile client process
        subprocess.run(
            ["terracotta", "connect", "localhost", "--port", "5100"], check=True
        )
