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
 #   MOMA: Mobility maps and optimal routes generator for ground vehicles    #
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
    """Simple interface to the Terracotta tile server"""

    def __init__(self, input_path, output_path, database_path):
        """Initialises the TileServer.

        Args:
            input_path (str): Path to the input directory containing raster files.
            output_path (str): Path to the output directory for optimized raster files.
            database_path (str): Path to the database directory for storing optimized raster files.
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
        """Updates the tile server by optimizing and ingesting raster files.

        This method runs the Terracotta raster optimization and database update processes.
        """

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
        """Runs the Terracotta XYZ tile server."""

        subprocess.run(
            [
                "terracotta",
                "serve",
                "--database",
                f"{self.database_path}/moma.sqlite",
                "--port",
                "5100",
            ],
            check=True,
        )

    def connect(self):
        """Connects to the Terracotta XYZ tile server."""

        subprocess.run(
            ["terracotta", "connect", "localhost", "--port", "5100"], check=True
        )
