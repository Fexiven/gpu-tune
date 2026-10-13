{
  description = "gpu-tune: automatic overclock / undervolt finder for NVIDIA GPUs on Linux";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
      # NixOS: the Vulkan loader comes from nixpkgs, the NVIDIA driver (libnvidia-ml, Vulkan ICD)
      # from /run/opengl-driver
      libPath = pkgs: pkgs.lib.makeLibraryPath [ pkgs.vulkan-loader ] + ":/run/opengl-driver/lib";
    in
    {
      packages = forAll (pkgs: {
        default = pkgs.rustPlatform.buildRustPackage {
          pname = "gpu-tune";
          version = "0.4.0";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;
          nativeBuildInputs = [ pkgs.makeWrapper ];
          # the test suite runs against a simulated GPU, so it works in the build sandbox
          doCheck = true;
          postFixup = ''
            wrapProgram $out/bin/gpu-tune --prefix LD_LIBRARY_PATH : ${libPath pkgs}
          '';
          meta = {
            description = "Automatic overclock / undervolt finder for NVIDIA GPUs, tested with Vulkan loads like games";
            mainProgram = "gpu-tune";
            license = pkgs.lib.licenses.mit;
            platforms = pkgs.lib.platforms.linux;
          };
        };
      });

      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt ];
          LD_LIBRARY_PATH = libPath pkgs;
        };
      });
    };
}
