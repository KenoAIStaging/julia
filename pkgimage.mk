SRCDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILDDIR := .
JULIAHOME := $(SRCDIR)
include $(JULIAHOME)/Make.inc
include $(JULIAHOME)/stdlib/stdlib.mk

DEPOTDIR := $(build_prefix)/share/julia

# set some influential environment variables
export JULIA_DEPOT_PATH := $(shell echo $(call cygpath_w,$(DEPOTDIR)))
export JULIA_LOAD_PATH := @stdlib$(PATHSEP)$(shell echo $(call cygpath_w,$(JULIAHOME)/stdlib))
unexport JULIA_PROJECT :=
unexport JULIA_BINDIR :=

export JULIA_FALLBACK_REPL := true

default: release
release: $(BUILDDIR)/stdlib/release.image
debug: $(BUILDDIR)/stdlib/debug.image
all: release debug

$(DEPOTDIR)/compiled:
	mkdir -p $@

print-depot-path:
	@$(call PRINT_JULIA, $(call spawn,$(JULIA_EXECUTABLE)) --startup-file=no -e '@show Base.DEPOT_PATH')

$(BUILDDIR)/stdlib/%.image: $(JULIAHOME)/stdlib/Project.toml $(JULIAHOME)/stdlib/Manifest.toml $(INDEPENDENT_STDLIBS_SRCS) $(DEPOTDIR)/compiled
	@$(call PRINT_JULIA, JULIA_CPU_TARGET="sysimage" $(call spawn,$(JULIA_EXECUTABLE)) --startup-file=no -e \
		'Base.Precompilation.precompilepkgs(configs=[``=>Base.CacheFlags(debug_level=2, opt_level=3), ``=>Base.CacheFlags(check_bounds=1, debug_level=2, opt_level=3)]; strict=true)')
	touch $@

$(BUILDDIR)/stdlib/release.image: $(build_private_libdir)/sys.$(SHLIB_EXT)
$(BUILDDIR)/stdlib/debug.image: $(build_private_libdir)/sys-debug.$(SHLIB_EXT)

# ---- stdlib caches for the unified sysimage (unifiedir bootstrap stage) ----
# Structural twin of the stock cache build above, run under sys-unified so
# every pkgimage is compiled by the UnifiedIR compiler port baked into that
# image.  Caches land in the same $(DEPOTDIR)/compiled bundled depot and
# coexist with the stock set: the cache filename slug hashes
# JLOptions().image_file and loading validates the recorded Base build_id,
# so each sysimage only accepts its own caches.
# UNIFIED_CACHE_PKGS optionally names root packages; the driver then builds
# only their dependency closure (subset validation) and the stamp is not
# written, so a later full run is not masked.
release-unified: $(BUILDDIR)/stdlib/release-unified.image
debug-unified: $(BUILDDIR)/stdlib/debug-unified.image

$(BUILDDIR)/stdlib/release-unified.image: UNIFIED_SYS := $(build_private_libdir)/sys-unified.$(SHLIB_EXT)
$(BUILDDIR)/stdlib/debug-unified.image: UNIFIED_SYS := $(build_private_libdir)/sys-unified-debug.$(SHLIB_EXT)

$(BUILDDIR)/stdlib/release-unified.image $(BUILDDIR)/stdlib/debug-unified.image: \
		$(JULIAHOME)/stdlib/Project.toml $(JULIAHOME)/stdlib/Manifest.toml \
		$(INDEPENDENT_STDLIBS_SRCS) $(JULIAHOME)/contrib/unified_stdlib_caches.jl | $(DEPOTDIR)/compiled
	@$(call PRINT_JULIA, JULIA_CPU_TARGET="sysimage" $(call spawn,$(JULIA_EXECUTABLE)) --sysimage $(call cygpath_w,$(UNIFIED_SYS)) --startup-file=no \
		$(call cygpath_w,$(JULIAHOME)/contrib/unified_stdlib_caches.jl) $(UNIFIED_CACHE_PKGS))
	$(if $(UNIFIED_CACHE_PKGS),@echo "unified-caches: subset build; stamp not written",touch $@)

$(BUILDDIR)/stdlib/release-unified.image: $(build_private_libdir)/sys-unified.$(SHLIB_EXT)
$(BUILDDIR)/stdlib/debug-unified.image: $(build_private_libdir)/sys-unified-debug.$(SHLIB_EXT)

clean:
	rm -rf $(DEPOTDIR)/compiled
	rm -f $(BUILDDIR)/stdlib/*.image
