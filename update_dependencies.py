#!/usr/bin/env python3
#
# This file is part of pyAMReX.
#
# License: BSD-3-Clause-LBNL

import argparse
import copy
import datetime
import json
import os
import sys
from pathlib import Path

import requests


def update(args):
    # list of repositories to update
    repo_dict = {}
    if args.all or args.amrex:
        repo_dict["amrex"] = {}
        repo_dict["amrex"]["commit"] = (
            "https://api.github.com/repos/AMReX-Codes/amrex/commits/development"
        )
        repo_dict["amrex"]["tags"] = (
            "https://api.github.com/repos/AMReX-Codes/amrex/tags"
        )
    if args.all or args.nanobind:
        repo_dict["nanobind"] = {}
        repo_dict["nanobind"]["commit"] = (
            "https://api.github.com/repos/wjakob/nanobind/commits/master"
        )
        repo_dict["nanobind"]["tags"] = (
            "https://api.github.com/repos/wjakob/nanobind/tags"
        )
    if args.all or args.pyamrex:
        repo_dict["pyamrex"] = {}
        repo_dict["pyamrex"]["commit"] = (
            "https://api.github.com/repos/AMReX-Codes/pyamrex/commits/development"
        )
        repo_dict["pyamrex"]["tags"] = (
            "https://api.github.com/repos/AMReX-Codes/pyamrex/tags"
        )

    # list of repositories labels for logging convenience
    repo_labels = {
        "amrex": "AMReX",
        "nanobind": "nanobind",
        "pyamrex": "pyAMReX",
    }

    # read from JSON file with dependencies data
    repo_dir = Path(__file__).parent.absolute()
    dependencies_file = os.path.join(repo_dir, "dependencies.json")
    try:
        with open(dependencies_file, "r") as file:
            dependencies_data = json.load(file)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit()

    # loop over repositories and update dependencies data
    for repo_name, repo_subdict in repo_dict.items():
        print(f"\nUpdating {repo_labels[repo_name]}...")

        # set keys to access dependencies data
        commit_key = f"commit_{repo_name}"
        version_key = f"version_{repo_name}"

        # get new commit information
        commit_response = requests.get(repo_subdict["commit"])
        commit_dict = commit_response.json()

        # set new commit
        repo_commit_sha = commit_dict["sha"]

        # get new version tag information
        tags_response = requests.get(repo_subdict["tags"])
        tags_list = tags_response.json()

        # filter out old-format tags for specific repositories
        tags_list_filtered = copy.deepcopy(tags_list)
        if repo_name == "amrex":
            tags_list_filtered = [
                tag_dict
                for tag_dict in tags_list
                if (tag_dict["name"] != "boxlib" and tag_dict["name"] != "v2024")
            ]

        # set new version tag
        if repo_name == "pyamrex":
            # current date version for the pyAMReX release update
            repo_version_tag = datetime.date.today().strftime("%y.%m")
        else:
            # latest available tag (index 0) for all other dependencies
            repo_version_tag = tags_list_filtered[0]["name"]

        # update commit
        if repo_name != "pyamrex":
            # use version tag instead of commit sha:
            # - for a release update
            # - for nanobind (always)
            # - if the commit has not changed since the last version tag
            use_version_tag = (
                args.release
                or (repo_name == "nanobind")
                or (repo_commit_sha == tags_list_filtered[0]["commit"]["sha"])
            )
            new_commit_sha = repo_version_tag if use_version_tag else repo_commit_sha
            print(f"- old commit: {dependencies_data[commit_key]}")
            print(f"- new commit: {new_commit_sha}")
            if dependencies_data[commit_key] == new_commit_sha:
                print("Skipping commit update...")
            else:
                print("Updating commit...")
                dependencies_data[f"commit_{repo_name}"] = new_commit_sha

        # update version
        if repo_name == "nanobind":
            print("Skipping version update... (minimum version set manually)")
        else:
            print(f"- old version: {dependencies_data[version_key]}")
            print(f"- new version: {repo_version_tag}")
            if dependencies_data[version_key] == repo_version_tag:
                print("Skipping version update...")
            else:
                print("Updating version...")
                dependencies_data[f"version_{repo_name}"] = repo_version_tag

    # write to JSON file with dependencies data
    with open(dependencies_file, "w") as file:
        json.dump(dependencies_data, file, indent=4)


if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser()

    # add arguments: AMReX option
    parser.add_argument(
        "--amrex",
        help="Update AMReX only",
        action="store_true",
        dest="amrex",
    )

    # add arguments: nanobind option
    parser.add_argument(
        "--nanobind",
        help="Update nanobind only",
        action="store_true",
        dest="nanobind",
    )

    # add arguments: pyAMReX option
    parser.add_argument(
        "--pyamrex",
        help="Update pyAMReX only",
        action="store_true",
        dest="pyamrex",
    )

    # add arguments: release option
    parser.add_argument(
        "--release",
        help="New release",
        action="store_true",
        dest="release",
    )

    # parse arguments
    args = parser.parse_args()

    # set args.all automatically
    args.all = False if (args.amrex or args.nanobind or args.pyamrex) else True

    # update
    update(args)
