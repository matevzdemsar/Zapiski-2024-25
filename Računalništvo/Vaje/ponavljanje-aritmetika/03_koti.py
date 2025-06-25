# =============================================================================
# Koti
# =====================================================================@005641=
# 1. podnaloga
# Predpostavite, da imate v spremenljivkah `stopinje1`, `minute1`,
# `sekunde1`, `stopinje2`, `minute2` in `sekunde2` shranjene stopinje,
# minute in sekunde dveh kotov (vse vrednosti so celoštevilske). 
# Sestavite program, ki bo ta dva kota seštel in izpisal niz oblike
# 
#     Vsota kotov je x stopinj, y minut in z sekund.
# 
# Seveda morata biti vrednosti spremenljivk `y` in `z` manjši od 60.
# Za vrednosti
# 
#     stopinje1 = 14
#     minute1 = 43
#     sekunde1 = 15
#     stopinje2 = 55
#     minute2 = 21
#     sekunde2 = 57
# 
# naj tako program izpiše
# 
#     Vsota kotov je 70 stopinj, 5 minut in 12 sekund.
# 
# Spremenljivk `stopinje1`, `minute1`, … ni potrebno definirati, saj so
# podane že v preambuli.
# =============================================================================
import math  # Število π je shranjeno v math.pi

stopinje1, minute1, sekunde1 = 14, 43, 15
stopinje2, minute2, sekunde2 = 55, 21, 57

kot = 2.185

ura, minuta = 17, 47
# =====================================================================@005642=
# 2. podnaloga
# Sestavite program, ki kot v radianih (realno število) pretvori in izpiše
# v stopinjah in minutah (kot celi števili; za minute naj program ne uporabi
# zaokroženja ampak naj samo odreže decimalna mesta). Predpostavite, da je
# kot podan v spremenljivki `kot`. 
# 
# Primer: Za 
# 
#     kot = 2.185
# 
# naj program izpiše
# 
#     2.185 radianov je 125 stopinj in 11 minut.
# 
# Spremenljivke `kot` ni potrebno definirati, saj je podana že v preambuli.
# =============================================================================

# =====================================================================@005643=
# 3. podnaloga
# Sestavite program, ki bo izračunal manjšega izmed kotov med urnim in
# minutnim kazalcem ob danem času. Čas je podan z uro in minutami.
# Upoštevajte, da se vsako minuto tudi urni kazalec malo prestavi.
# Trenutno uro in minuto imate podano v spremenljivkah z imenoma `ura`
# in `minuta`. Kot naj bo izpisan v obliki
#  
#     Kot med urnim in minutnim kazalcem je x stopinj in y minut.
# 
# kjer sta `x` in `y` celi števili (minut ne zaokrožuj, ampak samo odreži
# decimalni del).
# 
# Primer: za vrednosti
# 
#     ura = 17
#     minuta = 47
# 
# naj program izpiše
# 
#     Kot med urnim in minutnim kazalcem je 108 stopinj in 30 minut.
# =============================================================================






































































































# ============================================================================@
# fmt: off
"Če vam Python sporoča, da je v tej vrstici sintaktična napaka,"
"se napaka v resnici skriva v zadnjih vrsticah vaše kode."

"Kode od tu naprej NE SPREMINJAJTE!"

# isort: off
import json
import os
import re
import shutil
import sys
import traceback
import urllib.error
import urllib.request
import io
from contextlib import contextmanager


class VisibleStringIO(io.StringIO):
    def read(self, size=None):
        x = io.StringIO.read(self, size)
        print(x, end="")
        return x

    def readline(self, size=None):
        line = io.StringIO.readline(self, size)
        print(line, end="")
        return line


class TimeoutError(Exception):
    pass


class Check:
    parts = None
    current_part = None
    part_counter = None

    @staticmethod
    def has_solution(part):
        return part["solution"].strip() != ""

    @staticmethod
    def initialize(parts):
        Check.parts = parts
        for part in Check.parts:
            part["valid"] = True
            part["feedback"] = []
            part["secret"] = []

    @staticmethod
    def part():
        if Check.part_counter is None:
            Check.part_counter = 0
        else:
            Check.part_counter += 1
        Check.current_part = Check.parts[Check.part_counter]
        return Check.has_solution(Check.current_part)

    @staticmethod
    def feedback(message, *args, **kwargs):
        Check.current_part["feedback"].append(message.format(*args, **kwargs))

    @staticmethod
    def error(message, *args, **kwargs):
        Check.current_part["valid"] = False
        Check.feedback(message, *args, **kwargs)

    @staticmethod
    def clean(x, digits=6, typed=False):
        t = type(x)
        if t is float:
            x = round(x, digits)
            # Since -0.0 differs from 0.0 even after rounding,
            # we change it to 0.0 abusing the fact it behaves as False.
            v = x if x else 0.0
        elif t is complex:
            v = complex(
                Check.clean(x.real, digits, typed), Check.clean(x.imag, digits, typed)
            )
        elif t is list:
            v = list([Check.clean(y, digits, typed) for y in x])
        elif t is tuple:
            v = tuple([Check.clean(y, digits, typed) for y in x])
        elif t is dict:
            v = sorted(
                [
                    (Check.clean(k, digits, typed), Check.clean(v, digits, typed))
                    for (k, v) in x.items()
                ]
            )
        elif t is set:
            v = sorted([Check.clean(y, digits, typed) for y in x])
        else:
            v = x
        return (t, v) if typed else v

    @staticmethod
    def secret(x, hint=None, clean=None):
        clean = Check.get("clean", clean)
        Check.current_part["secret"].append((str(clean(x)), hint))

    @staticmethod
    def equal(expression, expected_result, clean=None, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        actual_result = eval(expression, global_env)
        if clean(actual_result) != clean(expected_result):
            Check.error(
                "Izraz {0} vrne {1!r} namesto {2!r}.",
                expression,
                actual_result,
                expected_result,
            )
            return False
        else:
            return True

    @staticmethod
    def approx(expression, expected_result, tol=1e-6, env=None, update_env=None):
        try:
            import numpy as np
        except ImportError:
            Check.error("Namestiti morate numpy.")
            return False
        if not isinstance(expected_result, np.ndarray):
            Check.error("Ta funkcija je namenjena testiranju za tip np.ndarray.")

        if env is None:
            env = dict()
        env.update({"np": np})
        global_env = Check.init_environment(env=env, update_env=update_env)
        actual_result = eval(expression, global_env)
        if type(actual_result) is not type(expected_result):
            Check.error(
                "Rezultat ima napačen tip. Pričakovan tip: {}, dobljen tip: {}.",
                type(expected_result).__name__,
                type(actual_result).__name__,
            )
            return False
        exp_shape = expected_result.shape
        act_shape = actual_result.shape
        if exp_shape != act_shape:
            Check.error(
                "Obliki se ne ujemata. Pričakovana oblika: {}, dobljena oblika: {}.",
                exp_shape,
                act_shape,
            )
            return False
        try:
            np.testing.assert_allclose(
                expected_result, actual_result, atol=tol, rtol=tol
            )
            return True
        except AssertionError as e:
            Check.error("Rezultat ni pravilen." + str(e))
            return False

    @staticmethod
    def run(statements, expected_state, clean=None, env=None, update_env=None):
        code = "\n".join(statements)
        statements = "  >>> " + "\n  >>> ".join(statements)
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        exec(code, global_env)
        errors = []
        for x, v in expected_state.items():
            if x not in global_env:
                errors.append(
                    "morajo nastaviti spremenljivko {0}, vendar je ne".format(x)
                )
            elif clean(global_env[x]) != clean(v):
                errors.append(
                    "nastavijo {0} na {1!r} namesto na {2!r}".format(
                        x, global_env[x], v
                    )
                )
        if errors:
            Check.error("Ukazi\n{0}\n{1}.", statements, ";\n".join(errors))
            return False
        else:
            return True

    @staticmethod
    @contextmanager
    def in_file(filename, content, encoding=None):
        encoding = Check.get("encoding", encoding)
        with open(filename, "w", encoding=encoding) as f:
            for line in content:
                print(line, file=f)
        old_feedback = Check.current_part["feedback"][:]
        yield
        new_feedback = Check.current_part["feedback"][len(old_feedback) :]
        Check.current_part["feedback"] = old_feedback
        if new_feedback:
            new_feedback = ["\n    ".join(error.split("\n")) for error in new_feedback]
            Check.error(
                "Pri vhodni datoteki {0} z vsebino\n  {1}\nso se pojavile naslednje napake:\n- {2}",
                filename,
                "\n  ".join(content),
                "\n- ".join(new_feedback),
            )

    @staticmethod
    @contextmanager
    def input(content, visible=None):
        old_stdin = sys.stdin
        old_feedback = Check.current_part["feedback"][:]
        try:
            with Check.set_stringio(visible):
                sys.stdin = Check.get("stringio")("\n".join(content) + "\n")
                yield
        finally:
            sys.stdin = old_stdin
        new_feedback = Check.current_part["feedback"][len(old_feedback) :]
        Check.current_part["feedback"] = old_feedback
        if new_feedback:
            new_feedback = ["\n  ".join(error.split("\n")) for error in new_feedback]
            Check.error(
                "Pri vhodu\n  {0}\nso se pojavile naslednje napake:\n- {1}",
                "\n  ".join(content),
                "\n- ".join(new_feedback),
            )

    @staticmethod
    def out_file(filename, content, encoding=None):
        encoding = Check.get("encoding", encoding)
        with open(filename, encoding=encoding) as f:
            out_lines = f.readlines()
        equal, diff, line_width = Check.difflines(out_lines, content)
        if equal:
            return True
        else:
            Check.error(
                "Izhodna datoteka {0}\n  je enaka{1}  namesto:\n  {2}",
                filename,
                (line_width - 7) * " ",
                "\n  ".join(diff),
            )
            return False

    @staticmethod
    def output(expression, content, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        too_many_read_requests = False
        try:
            exec(expression, global_env)
        except EOFError:
            too_many_read_requests = True
        finally:
            output = sys.stdout.getvalue().rstrip().splitlines()
            sys.stdout = old_stdout
        equal, diff, line_width = Check.difflines(output, content)
        if equal and not too_many_read_requests:
            return True
        else:
            if too_many_read_requests:
                Check.error("Program prevečkrat zahteva uporabnikov vnos.")
            if not equal:
                Check.error(
                    "Program izpiše{0}  namesto:\n  {1}",
                    (line_width - 13) * " ",
                    "\n  ".join(diff),
                )
            return False

    @staticmethod
    def difflines(actual_lines, expected_lines):
        actual_len, expected_len = len(actual_lines), len(expected_lines)
        if actual_len < expected_len:
            actual_lines += (expected_len - actual_len) * ["\n"]
        else:
            expected_lines += (actual_len - expected_len) * ["\n"]
        equal = True
        line_width = max(
            len(actual_line.rstrip())
            for actual_line in actual_lines + ["Program izpiše"]
        )
        diff = []
        for out, given in zip(actual_lines, expected_lines):
            out, given = out.rstrip(), given.rstrip()
            if out != given:
                equal = False
            diff.append(
                "{0} {1} {2}".format(
                    out.ljust(line_width), "|" if out == given else "*", given
                )
            )
        return equal, diff, line_width

    @staticmethod
    def init_environment(env=None, update_env=None):
        global_env = globals()
        if not Check.get("update_env", update_env):
            global_env = dict(global_env)
        global_env.update(Check.get("env", env))
        return global_env

    @staticmethod
    def generator(
        expression,
        expected_values,
        should_stop=None,
        further_iter=None,
        clean=None,
        env=None,
        update_env=None,
    ):
        from types import GeneratorType

        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        gen = eval(expression, global_env)
        if not isinstance(gen, GeneratorType):
            Check.error("Izraz {0} ni generator.", expression)
            return False

        try:
            for iteration, expected_value in enumerate(expected_values):
                actual_value = next(gen)
                if clean(actual_value) != clean(expected_value):
                    Check.error(
                        "Vrednost #{0}, ki jo vrne generator {1} je {2!r} namesto {3!r}.",
                        iteration,
                        expression,
                        actual_value,
                        expected_value,
                    )
                    return False
            for _ in range(Check.get("further_iter", further_iter)):
                next(gen)  # we will not validate it
        except StopIteration:
            Check.error("Generator {0} se prehitro izteče.", expression)
            return False

        if Check.get("should_stop", should_stop):
            try:
                next(gen)
                Check.error("Generator {0} se ne izteče (dovolj zgodaj).", expression)
            except StopIteration:
                pass  # this is fine
        return True

    @staticmethod
    def summarize():
        for i, part in enumerate(Check.parts):
            if not Check.has_solution(part):
                print("{0}. podnaloga je brez rešitve.".format(i + 1))
            elif not part["valid"]:
                print("{0}. podnaloga nima veljavne rešitve.".format(i + 1))
            else:
                print("{0}. podnaloga ima veljavno rešitev.".format(i + 1))
            for message in part["feedback"]:
                print("  - {0}".format("\n    ".join(message.splitlines())))

    settings_stack = [
        {
            "clean": clean.__func__,
            "encoding": None,
            "env": {},
            "further_iter": 0,
            "should_stop": False,
            "stringio": VisibleStringIO,
            "update_env": False,
        }
    ]

    @staticmethod
    def get(key, value=None):
        if value is None:
            return Check.settings_stack[-1][key]
        return value

    @staticmethod
    @contextmanager
    def set(**kwargs):
        settings = dict(Check.settings_stack[-1])
        settings.update(kwargs)
        Check.settings_stack.append(settings)
        try:
            yield
        finally:
            Check.settings_stack.pop()

    @staticmethod
    @contextmanager
    def set_clean(clean=None, **kwargs):
        clean = clean or Check.clean
        with Check.set(clean=(lambda x: clean(x, **kwargs)) if kwargs else clean):
            yield

    @staticmethod
    @contextmanager
    def set_environment(**kwargs):
        env = dict(Check.get("env"))
        env.update(kwargs)
        with Check.set(env=env):
            yield

    @staticmethod
    @contextmanager
    def set_stringio(stringio):
        if stringio is True:
            stringio = VisibleStringIO
        elif stringio is False:
            stringio = io.StringIO
        if stringio is None or stringio is Check.get("stringio"):
            yield
        else:
            with Check.set(stringio=stringio):
                yield

    @staticmethod
    @contextmanager
    def time_limit(timeout_seconds=1):
        from signal import SIGINT, raise_signal
        from threading import Timer

        def interrupt_main():
            raise_signal(SIGINT)

        timer = Timer(timeout_seconds, interrupt_main)
        timer.start()
        try:
            yield
        except KeyboardInterrupt:
            raise TimeoutError
        finally:
            timer.cancel()


def _validate_current_file():
    def extract_parts(filename):
        with open(filename, encoding="utf-8") as f:
            source = f.read()
        part_regex = re.compile(
            r"# =+@(?P<part>\d+)=\s*\n"  # beginning of header
            r"(\s*#( [^\n]*)?\n)+?"  # description
            r"\s*# =+\s*?\n"  # end of header
            r"(?P<solution>.*?)"  # solution
            r"(?=\n\s*# =+@)",  # beginning of next part
            flags=re.DOTALL | re.MULTILINE,
        )
        parts = [
            {"part": int(match.group("part")), "solution": match.group("solution")}
            for match in part_regex.finditer(source)
        ]
        # The last solution extends all the way to the validation code,
        # so we strip any trailing whitespace from it.
        parts[-1]["solution"] = parts[-1]["solution"].rstrip()
        return parts

    def backup(filename):
        backup_filename = None
        suffix = 1
        while not backup_filename or os.path.exists(backup_filename):
            backup_filename = "{0}.{1}".format(filename, suffix)
            suffix += 1
        shutil.copy(filename, backup_filename)
        return backup_filename

    def submit_parts(parts, url, token):
        submitted_parts = []
        for part in parts:
            if Check.has_solution(part):
                submitted_part = {
                    "part": part["part"],
                    "solution": part["solution"],
                    "valid": part["valid"],
                    "secret": [x for (x, _) in part["secret"]],
                    "feedback": json.dumps(part["feedback"]),
                }
                if "token" in part:
                    submitted_part["token"] = part["token"]
                submitted_parts.append(submitted_part)
        data = json.dumps(submitted_parts).encode("utf-8")
        headers = {"Authorization": token, "content-type": "application/json"}
        request = urllib.request.Request(url, data=data, headers=headers)
        # This is a workaround because some clients (and not macOS ones!) report
        # <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1129)>
        import ssl

        context = ssl._create_unverified_context()
        response = urllib.request.urlopen(request, context=context)
        # When the issue is resolved, the following should be used
        # response = urllib.request.urlopen(request)
        return json.loads(response.read().decode("utf-8"))

    def update_attempts(old_parts, response):
        updates = {}
        for part in response["attempts"]:
            part["feedback"] = json.loads(part["feedback"])
            updates[part["part"]] = part
        for part in old_parts:
            valid_before = part["valid"]
            part.update(updates.get(part["part"], {}))
            valid_after = part["valid"]
            if valid_before and not valid_after:
                wrong_index = response["wrong_indices"].get(str(part["part"]))
                if wrong_index is not None:
                    hint = part["secret"][wrong_index][1]
                    if hint:
                        part["feedback"].append("Namig: {}".format(hint))

    filename = os.path.abspath(sys.argv[0])
    file_parts = extract_parts(filename)
    Check.initialize(file_parts)

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1NjQxLCJ1c2VyIjo5MTQyfQ:1to0Kc:cWThVyxARKLO5gGxryLsF3qiXrqWHlbRckLCV6w1V3M"
        try:
            konec = False
            for s1 in [5, 10, 12, 15, 20, 22, 27, 30]:
                if konec: break
                for s2 in [5, 12, 19, 27, 30]:
                    if konec: break
                    for m1 in [35, 39, 40, 45, 55, 59]:
                        if konec: break
                        for m2 in [30, 42, 51, 59]:
                            if konec: break
                            for sek1 in [35, 37, 45, 51, 55, 59]:
                                if konec: break
                                for sek2 in [30, 37, 42, 51, 59]:
                                    vhod = {"stopinje1": s1, "stopinje2": s2, "minute1": m1, "minute2": m2, "sekunde1": sek1, "sekunde2": sek2}
                                    st, mins, sek = vhod["stopinje1"] + vhod["stopinje2"], vhod["minute1"] + vhod["minute2"], vhod["sekunde1"] + vhod["sekunde2"]
                                    st, mins, sek = st + (mins + sek // 60) // 60, (mins + sek // 60) % 60, sek % 60
                                    exec_info = Check.execute(Check.current['solution'], vhod)
                                    okolje, izpis = exec_info['env'], exec_info['stdout'].rstrip()
                                    if izpis != "Vsota kotov je {0} stopinj, {1} minut in {2} sekund.".format(st, mins, sek):
                                        Check.error("Za vrednosti ({0}, {1}, {2}) in ({3}, {4}, {5}) je izpis napačen. Vaš izpis:\n{6}".format(
                                            vhod["stopinje1"], vhod["minute1"], vhod["sekunde1"], vhod["stopinje2"], vhod["minute2"], vhod["sekunde2"],
                                            '<<<prazno>>>' if len(izpis) == 0 else izpis))
                                        konec = True
                                        break
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1NjQyLCJ1c2VyIjo5MTQyfQ:1to0Kc:AD1el5IoDl0zjOxr7IHvCmx5yCFexHaVitzi6Q48tlQ"
        try:
            import math
            testi = [8.409771825745423, 8.155515067296895, 3.726267425218536, 1.8766818502911842, 9.550079791032239, 0.0028909653434028293,
                     2.9759100341730047, 2.9291119223891404, 4.194144719019944, 4.499573109009985, 8.041658594891604, 0.4395642199422767,
                     5.635175229025035, 2.552312072535443, 3.1839727256580974, 3.282690303544401, 8.507473209416982, 8.328826316108337,
                     3.3004266898822108, 1.7077919561152521, 1.2657650827451161, 3.775267342987835, 1.390913615882241, 7.1467298258180385,
                     4.322705731306612, 6.903562078944439, 3.4434441442713615, 0.6447892751694018, 6.489359744985959, 5.325037262435864,
                     8.586756395693932, 6.638404130999566, 0.1034784094086294, 4.5433606928620005, 3.9686641035838788, 1.0955781341510784,
                     5.920503721062678, 8.083543356580122, 5.183923337092015, 3.107450085803979, 4.260329557854789, 7.51312487055676,
                     7.465283447093065, 3.359137476113615, 9.310657446488577, 1.0342760592590128, 3.9760135290091103, 6.365736205867818,
                     1.3711314699478072, 3.6553162361055955]
            for kot in testi:
                vhod = {'kot': kot}
                stopinje = kot * 360 / (2 * math.pi)
                stopinje, minute = int(stopinje), int((stopinje - int(stopinje))*60)
                exec_info = Check.execute(Check.current['solution'], vhod)
                okolje, izpis = exec_info['env'], exec_info['stdout'].rstrip()
                if izpis != "{0} radianov je {1} stopinj in {2} minut.".format(kot, stopinje, minute):
                    Check.error("Za kot = {0} je izpis napačne oblike. Vaš izpis:\n{1}".format(kot, '<<<prazno>>>' if len(izpis) == 0 else izpis))
                    break
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1NjQzLCJ1c2VyIjo5MTQyfQ:1to0Kc:lDitFBzkbX9WISuztxiHAnj_z5QKzPmkQvcBSHjS_xU"
        try:
            konec = False
            for ura in range(0, 23+1):
                if konec: break
                for minuta in range(0, 59+1):   
                    vhod = {"ura": ura, "minuta": minuta}
                    exec_info = Check.execute(Check.current['solution'], vhod)
                    okolje, izpis = exec_info['env'], exec_info['stdout'].rstrip()
                    ura = ura % 12
                    alfa = 60 * (6 * minuta)
                    beta = 60 * (30*ura + 0.5*minuta)
                    minute = abs(alfa - beta)
                    minute = min(minute, 360*60 - minute)
                    minute = int(minute)
                    stopinje = minute // 60
                    minute = minute % 60
                    pravi_izpis = 'Kot med urnim in minutnim kazalcem je {0} stopinj in {1} minut.'.format(stopinje, minute)
                    if izpis != pravi_izpis:
                        Check.error("Pri podatkih ura = {0} in minuta = {1} izpis ni pravilen. Vaš izpis:\n{2}".format(ura, minuta,
                            '<<<prazno>>>' if len(izpis) == 0 else izpis))
                        konec = True
                        break
        except TimeoutError:
            Check.error("Dovoljen čas izvajanja presežen")
        except Exception:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    print("Shranjujem rešitve na strežnik... ", end="")
    try:
        url = "https://www.projekt-tomo.si/api/attempts/submit/"
        token = "Token 46d0a66581ff1387631370d37873b8d1806bc918"
        response = submit_parts(Check.parts, url, token)
    except urllib.error.URLError:
        message = (
            "\n"
            "-------------------------------------------------------------------\n"
            "PRI SHRANJEVANJU JE PRIŠLO DO NAPAKE!\n"
            "Preberite napako in poskusite znova ali se posvetujte z asistentom.\n"
            "-------------------------------------------------------------------\n"
        )
        print(message)
        traceback.print_exc()
        print(message)
        sys.exit(1)
    else:
        print("Rešitve so shranjene.")
        update_attempts(Check.parts, response)
        if "update" in response:
            print("Updating file... ", end="")
            backup_filename = backup(filename)
            with open(__file__, "w", encoding="utf-8") as f:
                f.write(response["update"])
            print("Previous file has been renamed to {0}.".format(backup_filename))
            print("If the file did not refresh in your editor, close and reopen it.")
    Check.summarize()


if __name__ == "__main__":
    _validate_current_file()
